use alloc::vec::Vec;
use core::cmp::max;
use core::ops::Deref;

use itertools::{Itertools, chain};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_field::Field;
use tracing::instrument;

use crate::{Interaction, SymbolicAirBuilder, SymbolicExpression};

#[derive(Clone, Debug)]
pub struct AirMeta {
    pub width: usize,
    pub public_value_count: usize,
    pub constraint_count: usize,
    pub zero_check_uv_degree: usize,
    pub zero_check_mv_degree: usize,
    pub interaction_count: usize,
    pub eval_check_uv_degree: usize,
    pub eval_check_mv_degree: usize,
    pub max_bus_index: usize,
    pub max_field_count: usize,
}

#[derive(Clone)]
pub struct VerifyingKey {
    pub(crate) metas: Vec<AirMeta>,
}

pub struct ProvingKey {
    vk: VerifyingKey,
}

impl AirMeta {
    pub fn alpha_power_count(&self) -> usize {
        self.constraint_count + 2 * self.interaction_count
    }

    pub fn has_interaction(&self) -> bool {
        self.interaction_count != 0
    }

    pub fn regular_sumcheck_degree(&self) -> usize {
        max(self.zero_check_mv_degree, self.eval_check_mv_degree)
    }
}

impl VerifyingKey {
    pub fn metas(&self) -> &[AirMeta] {
        &self.metas
    }

    pub fn max_width(&self) -> usize {
        itertools::max(self.metas.iter().map(|meta| meta.width)).unwrap_or_default()
    }

    pub fn max_alpha_power_count(&self) -> usize {
        itertools::max(self.metas.iter().map(|meta| meta.alpha_power_count())).unwrap_or_default()
    }

    pub fn max_regular_sumcheck_degree(&self) -> usize {
        itertools::max(self.metas.iter().map(AirMeta::regular_sumcheck_degree)).unwrap_or_default()
    }

    pub fn has_any_interaction(&self) -> bool {
        self.metas.iter().any(AirMeta::has_interaction)
    }

    pub fn max_bus_index(&self) -> usize {
        itertools::max(self.metas.iter().map(|meta| meta.max_bus_index)).unwrap_or_default()
    }

    pub fn max_field_count(&self) -> usize {
        itertools::max(self.metas.iter().map(|meta| meta.max_field_count)).unwrap_or_default()
    }
}

impl Deref for ProvingKey {
    type Target = VerifyingKey;

    fn deref(&self) -> &Self::Target {
        &self.vk
    }
}

pub fn keygen_vk<'a, Val, A>(airs: impl IntoIterator<Item = &'a A>) -> VerifyingKey
where
    Val: Field,
    A: 'a + BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>,
{
    let metas = airs
        .into_iter()
        .map(|air| {
            let width = air.width();
            let public_value_count = air.num_public_values();
            let (_, _, zero_check_uv_degree, eval_check_uv_degree) =
                get_symbolic_info(air, 0, 0, public_value_count);
            let (constraints, interactions, zero_check_mv_degree, eval_check_mv_degree) =
                get_symbolic_info(air, 1, 0, public_value_count);
            let constraint_count = constraints.len();
            let interaction_count = interactions.len();
            let max_bus_index =
                itertools::max(interactions.iter().map(|i| i.bus_index)).unwrap_or_default();
            let max_field_count =
                itertools::max(interactions.iter().map(|i| i.fields.len())).unwrap_or_default();
            AirMeta {
                width,
                public_value_count,
                zero_check_uv_degree,
                zero_check_mv_degree,
                constraint_count,
                eval_check_uv_degree,
                eval_check_mv_degree,
                interaction_count,
                max_bus_index,
                max_field_count,
            }
        })
        .collect();
    VerifyingKey { metas }
}

pub fn keygen_pk<'a, Val, A>(vk: &VerifyingKey, _: impl IntoIterator<Item = &'a A>) -> ProvingKey
where
    Val: Field,
    A: 'a + BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>,
{
    ProvingKey { vk: vk.clone() }
}

pub fn keygen<'a, Val, A>(airs: impl IntoIterator<Item = &'a A>) -> (VerifyingKey, ProvingKey)
where
    Val: Field,
    A: 'a + BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>,
{
    let airs = airs.into_iter().collect_vec();
    let vk = keygen_vk(airs.clone());
    let pk = keygen_pk(&vk, airs);
    (vk, pk)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
#[allow(clippy::type_complexity)]
fn get_symbolic_info<F, A>(
    air: &A,
    is_transition_degree: usize,
    preprocessed_width: usize,
    num_public_values: usize,
) -> (
    Vec<SymbolicExpression<F>>,
    Vec<Interaction<SymbolicExpression<F>>>,
    usize,
    usize,
)
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(
        is_transition_degree,
        preprocessed_width,
        air.width(),
        num_public_values,
    );
    air.eval(&mut builder);
    let (constraints, interactions) = builder.into_symbolic_constraints();
    let zero_check_degree = max_degree(&constraints);
    let eval_check_degree = max_degree(
        interactions
            .iter()
            .flat_map(|interaction| chain![&interaction.fields, [&interaction.count]]),
    );
    (
        constraints,
        interactions,
        zero_check_degree,
        eval_check_degree,
    )
}

fn max_degree<'a, F: 'a>(exprs: impl IntoIterator<Item = &'a SymbolicExpression<F>>) -> usize {
    itertools::max(exprs.into_iter().map(SymbolicExpression::degree_multiple)).unwrap_or_default()
}
