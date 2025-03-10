use alloc::vec;
use alloc::vec::Vec;
use core::cmp::max;

use itertools::{Itertools, chain};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::SymbolicVariable;
use crate::{Entry, Interaction, InteractionAirBuilder, InteractionType};

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree =
        get_max_constraint_degree(air, preprocessed_width, num_public_values).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    max_degree(&get_symbolic_constraints(air, preprocessed_width, num_public_values).0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> (
    Vec<SymbolicExpression<F>>,
    Vec<Interaction<SymbolicExpression<F>>>,
)
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(preprocessed_width, air.width(), num_public_values);
    air.eval(&mut builder);
    (builder.constraints, builder.interactions)
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
    interactions: Vec<Interaction<SymbolicExpression<F>>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub(crate) fn new(preprocessed_width: usize, width: usize, num_public_values: usize) -> Self {
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
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            constraints: vec![],
            interactions: vec![],
        }
    }
}

pub(crate) fn max_degree<F>(exprs: &[SymbolicExpression<F>]) -> usize {
    itertools::max(exprs.iter().map(SymbolicExpression::degree_multiple)).unwrap_or(0)
}

pub(crate) fn interaction_chunks<F>(
    max_constraint_degree: usize,
    interactions: &[Interaction<SymbolicExpression<F>>],
) -> Vec<Vec<usize>> {
    if interactions.is_empty() {
        return Vec::new();
    }

    let interaction_degrees = interactions
        .iter()
        .map(|i| (max_degree(&i.fields), i.count.degree_multiple()))
        .collect_vec();

    let mut chunks = vec![vec![]];
    let mut current_numer_degree = 0;
    let mut current_denom_degree = 0;
    interaction_degrees.into_iter().enumerate().for_each(
        |(idx, (field_degree, count_degree))| {
            current_numer_degree = max(
                current_numer_degree + field_degree,
                current_denom_degree + count_degree,
            );
            current_denom_degree += field_degree;
            if max(current_numer_degree, current_denom_degree + 1) <= max_constraint_degree {
                chunks.last_mut().unwrap().push(idx);
            } else {
                chunks.push(vec![idx]);
                current_numer_degree = count_degree;
                current_denom_degree = field_degree;
                if max(current_numer_degree, current_denom_degree + 1) > max_constraint_degree {
                    panic!("Interaction with field_degree={field_degree}, count_degree={count_degree} exceeds max_constraint_degree={max_constraint_degree}")
                }
            }
        },
    );
    chunks
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
            SymbolicExpression::IsTransition
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
