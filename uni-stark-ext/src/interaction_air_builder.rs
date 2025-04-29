use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

use crate::{InteractionAirBuilder, InteractionType, StarkGenericConfig, Val};

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

pub struct ProverInteractionFolder<'a, SC: StarkGenericConfig> {
    pub main: ViewPair<'a, Val<SC>>,
    pub public_values: &'a Vec<Val<SC>>,
    pub beta_powers: &'a [SC::Challenge],
    pub gamma_powers: &'a [SC::Challenge],
    pub numers: &'a mut [Val<SC>],
    pub denoms: &'a mut [SC::Challenge],
    pub interaction_index: usize,
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverInteractionFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = Val<SC>;
    type Var = Val<SC>;
    type M = ViewPair<'a, Val<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        unimplemented!()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        unimplemented!()
    }

    #[inline]
    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unimplemented!()
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, _: I) {}
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for ProverInteractionFolder<'_, SC> {
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> InteractionAirBuilder for ProverInteractionFolder<'_, SC> {
    const ONLY_INTERACTION: bool = true;

    #[inline]
    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let mut count = count.into();
        if interaction_type == InteractionType::Receive {
            count = -count;
        }
        self.numers[self.interaction_index] = count;

        let mut fields = fields.into_iter();
        self.denoms[self.interaction_index] =
            self.gamma_powers[bus_index] + fields.next().unwrap().into();
        izip!(fields, self.beta_powers).for_each(|(field, beta_power)| {
            self.denoms[self.interaction_index] += *beta_power * field.into();
        });

        self.interaction_index += 1;
    }
}

#[inline]
pub(crate) fn eval_log_up<AB: ExtensionBuilder>(
    builder: &mut AB,
    interaction_chunks: &[Vec<usize>],
    numers: &[AB::Expr],
    denoms: &[AB::ExprEF],
    local: &[AB::VarEF],
    next: &[AB::VarEF],
    sum: AB::ExprEF,
) {
    let chunk_local = &local[..interaction_chunks.len()];
    let chunk_next = &next[..interaction_chunks.len()];
    let sum_local = local[interaction_chunks.len()];
    let sum_next = next[interaction_chunks.len()];

    izip!(chunk_local, interaction_chunks).for_each(|(chunk_sum, chunk)| {
        let lhs = chunk
            .iter()
            .fold((*chunk_sum).into(), |acc, i| acc * denoms[*i].clone());
        let rhs = if chunk.len() == 1 {
            numers[chunk[0]].clone().into()
        } else {
            chunk
                .iter()
                .map(|i| {
                    chunk
                        .iter()
                        .filter(|j| i != *j)
                        .map(|j| denoms[*j].clone())
                        .product::<AB::ExprEF>()
                        * numers[*i].clone()
                })
                .sum::<AB::ExprEF>()
        };

        builder.assert_eq_ext(lhs, rhs);
    });

    builder.when_transition().assert_eq_ext(
        sum_next.into() - sum_local.into(),
        chunk_next.iter().copied().map_into().sum::<AB::ExprEF>(),
    );
    builder.when_first_row().assert_eq_ext(
        sum_local,
        chunk_local.iter().copied().map_into().sum::<AB::ExprEF>(),
    );
    builder.when_last_row().assert_eq_ext(sum_local, sum);
}
