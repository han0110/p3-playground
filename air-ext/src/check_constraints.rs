use alloc::collections::btree_map::BTreeMap;
use alloc::format;
use alloc::vec::Vec;

use hashbrown::hash_map::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use tracing::instrument;

use crate::{InteractionBuilder, InteractionType, ProverInput};

#[instrument(name = "check constraints", skip_all)]
pub fn check_constraints<F, A>(prover_inputs: &[ProverInput<F, A>])
where
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, F>>,
{
    let mut interactions = BTreeMap::new();

    prover_inputs
        .iter()
        .enumerate()
        .for_each(|(air_index, input)| {
            let height = input.trace.height();

            (0..height).for_each(|i| {
                let i_next = (i + 1) % height;

                let local = input.trace.row_slice(i).unwrap();
                let next = input.trace.row_slice(i_next).unwrap();
                let main = VerticalPair::new(
                    RowMajorMatrixView::new_row(&*local),
                    RowMajorMatrixView::new_row(&*next),
                );

                let mut builder = DebugConstraintBuilder {
                    air_index,
                    row_index: i,
                    main,
                    public_values: &input.public_values,
                    is_first_row: F::from_bool(i == 0),
                    is_last_row: F::from_bool(i == height - 1),
                    is_transition: F::from_bool(i != height - 1),
                    interactions: &mut interactions,
                };

                input.air.eval(&mut builder);
            });
        });

    let mut imbalances = Vec::new();
    interactions
        .into_iter()
        .for_each(|(bus_index, interactions)| {
            interactions.into_iter().for_each(|(fields, records)| {
                let mut sum = F::ZERO;
                records
                    .iter()
                    .for_each(|(_, interaction_type, count)| match interaction_type {
                        InteractionType::Send => sum += *count,
                        InteractionType::Receive => sum -= *count,
                    });
                if !sum.is_zero() {
                    imbalances.push(
                        format!("Bus {bus_index} failed to balance the multiplicities for fields: {fields:?}. The bus records for this were:")
                    );
                    imbalances.extend(
                        records
                            .into_iter()
                            .map(|(air_index, interaction_type, count)| {
                                format!("  Air index: {air_index}, interaction type: {interaction_type:?}, count: {count:?}")
                            }),
                    );
                }
            })
        });
    if !imbalances.is_empty() {
        panic!(
            "Interaction multiset equality check failed.\n{}",
            imbalances.iter().join("\n")
        );
    }
}

/// An `AirBuilder` which asserts that each constraint is zero, allowing any failed constraints to
/// be detected early.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field> {
    air_index: usize,
    row_index: usize,
    main: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    public_values: &'a [F],
    is_first_row: F,
    is_last_row: F,
    is_transition: F,
    #[allow(clippy::type_complexity)]
    interactions: &'a mut BTreeMap<usize, HashMap<Vec<F>, Vec<(usize, InteractionType, F)>>>,
}

impl<'a, F> AirBuilder for DebugConstraintBuilder<'a, F>
where
    F: Field,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert_eq!(
            x.into(),
            F::ZERO,
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }
}

impl<F: Field> AirBuilderWithPublicValues for DebugConstraintBuilder<'_, F> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<F: Field> InteractionBuilder for DebugConstraintBuilder<'_, F> {
    const ONLY_INTERACTION: bool = false;

    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        self.interactions
            .entry(bus_index)
            .or_default()
            .entry(fields.into_iter().map_into().collect())
            .or_default()
            .push((self.air_index, interaction_type, count.into()));
    }
}
