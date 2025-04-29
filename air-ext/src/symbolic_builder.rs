use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, chain};
use p3_air::{AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::SymbolicVariable;
use crate::{Entry, Interaction, InteractionAirBuilder, InteractionType};

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    is_transition_degree: usize,
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
    interactions: Vec<Interaction<SymbolicExpression<F>>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub fn new(
        is_transition_degree: usize,
        preprocessed_width: usize,
        width: usize,
        num_public_values: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        Self {
            is_transition_degree,
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            constraints: vec![],
            interactions: vec![],
        }
    }

    pub fn into_symbolic_constraints(
        self,
    ) -> (
        Vec<SymbolicExpression<F>>,
        Vec<Interaction<SymbolicExpression<F>>>,
    ) {
        (self.constraints, self.interactions)
    }
}

impl<F: Field> AirBuilder for SymbolicAirBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition {
                degree: self.is_transition_degree,
            }
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

impl<F: Field> AirBuilderWithPublicValues for SymbolicAirBuilder<F> {
    type PublicVar = SymbolicVariable<F>;
    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field> PairBuilder for SymbolicAirBuilder<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field> InteractionAirBuilder for SymbolicAirBuilder<F> {
    const ONLY_INTERACTION: bool = false;

    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let fields = fields.into_iter().map_into().collect_vec();
        let count = count.into();
        assert!(chain![&fields, [&count]].all(|expr| !expr.has_selector()));
        self.interactions.push(Interaction {
            fields,
            count,
            bus_index,
            interaction_type,
        })
    }
}
