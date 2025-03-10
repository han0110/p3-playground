use alloc::vec;
use alloc::vec::Vec;
use core::cmp::max;
use core::ops::Deref;

use itertools::Itertools;
use p3_air::{Air, BaseAirWithPublicValues};
use p3_field::{BasedVectorSpace, Field};
use p3_util::log2_ceil_usize;

use crate::{
    Interaction, ProofPerAir, StarkGenericConfig, SymbolicAirBuilder, SymbolicExpression, Val,
    get_symbolic_constraints, max_degree,
};

#[derive(Clone)]
pub struct VerifyingKeyPerAir {
    pub width: usize,
    pub log_quotient_degree: usize,
    pub constraint_count: usize,
    pub interaction_count: usize,
    pub interaction_chunks: Vec<Vec<usize>>,
    pub max_bus_index: usize,
    pub max_field_count: usize,
}

#[derive(Clone)]
pub struct VerifyingKey {
    pub(crate) per_air: Vec<VerifyingKeyPerAir>,
}

pub struct ProvingKey {
    vk: VerifyingKey,
}

impl VerifyingKeyPerAir {
    pub fn quotient_degree(&self) -> usize {
        1 << self.log_quotient_degree
    }

    pub fn has_interaction(&self) -> bool {
        self.interaction_count != 0
    }

    pub fn valid_shape<SC: StarkGenericConfig>(&self, proof: &ProofPerAir<SC>) -> bool {
        let ProofPerAir {
            log_up_sum,
            opened_values,
            ..
        } = proof;
        let dimension = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        let log_up_width = self
            .has_interaction()
            .then(|| self.interaction_chunks.len() + 1)
            .unwrap_or_default();
        self.has_interaction() == log_up_sum.is_some()
            && opened_values.main_local.len() == self.width
            && opened_values.main_next.len() == self.width
            && opened_values.log_up_local.len() == dimension * log_up_width
            && opened_values.log_up_next.len() == dimension * log_up_width
            && opened_values.quotient_chunks.len() == self.quotient_degree()
            && opened_values
                .quotient_chunks
                .iter()
                .all(|qc| qc.len() == dimension)
    }
}

impl VerifyingKey {
    pub fn per_air(&self) -> &[VerifyingKeyPerAir] {
        &self.per_air
    }

    pub fn has_any_interaction(&self) -> bool {
        self.per_air.iter().any(VerifyingKeyPerAir::has_interaction)
    }

    pub fn max_bus_index(&self) -> usize {
        itertools::max(self.per_air.iter().map(|input| input.max_bus_index)).unwrap_or_default()
    }

    pub fn max_field_count(&self) -> usize {
        itertools::max(self.per_air.iter().map(|input| input.max_field_count)).unwrap_or_default()
    }

    pub fn max_constraint_count(&self) -> usize {
        itertools::max(self.per_air.iter().map(|input| input.constraint_count)).unwrap_or_default()
    }
}

impl Deref for ProvingKey {
    type Target = VerifyingKey;

    fn deref(&self) -> &Self::Target {
        &self.vk
    }
}

pub fn keygen_vk<'a, Val, A>(
    max_constraint_degree: usize,
    airs: impl IntoIterator<Item = &'a A>,
) -> VerifyingKey
where
    Val: Field,
    A: 'a + BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>,
{
    let per_air = airs
        .into_iter()
        .map(|air| {
            let width = air.width();
            let (constraints, interactions) =
                get_symbolic_constraints(air, 0, air.num_public_values());
            let constraint_degree = max_degree(&constraints);
            assert!(
                constraint_degree <= max_constraint_degree,
                "Max constraint degree {max_constraint_degree}, but got {constraint_degree}"
            );
            let log_quotient_degree = log2_ceil_usize(constraint_degree.saturating_sub(1));
            let interaction_chunks =
                interaction_chunks((1 << log_quotient_degree) + 1, &interactions);
            let log_up_constraint_count = if interactions.is_empty() {
                0
            } else {
                interaction_chunks.len() + 3
            };
            let constraint_count = constraints.len() + log_up_constraint_count;
            let interaction_count = interactions.len();
            let max_bus_index =
                itertools::max(interactions.iter().map(|i| i.bus_index)).unwrap_or_default();
            let max_field_count =
                itertools::max(interactions.iter().map(|i| i.fields.len())).unwrap_or_default();
            VerifyingKeyPerAir {
                width,
                log_quotient_degree,
                constraint_count,
                interaction_count,
                interaction_chunks,
                max_bus_index,
                max_field_count,
            }
        })
        .collect();
    VerifyingKey { per_air }
}

pub fn keygen_pk<'a, Val, A>(vk: &VerifyingKey, _: impl IntoIterator<Item = &'a A>) -> ProvingKey
where
    Val: Field,
    A: 'a + BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>,
{
    ProvingKey { vk: vk.clone() }
}

pub fn keygen<'a, Val, A>(
    max_constraint_degree: usize,
    airs: impl IntoIterator<Item = &'a A>,
) -> (VerifyingKey, ProvingKey)
where
    Val: Field,
    A: 'a + BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>,
{
    let airs = airs.into_iter().collect_vec();
    let vk = keygen_vk(max_constraint_degree, airs.clone());
    let pk = keygen_pk(&vk, airs);
    (vk, pk)
}

fn interaction_chunks<F>(
    max_constraint_degree: usize,
    interactions: &[Interaction<SymbolicExpression<F>>],
) -> Vec<Vec<usize>> {
    if interactions.is_empty() {
        return Vec::new();
    }

    let interaction_degrees = interactions
        .iter()
        .enumerate()
        .map(|(idx, i)| (idx, max_degree(&i.fields), i.count.degree_multiple()))
        .sorted_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)))
        .collect_vec();

    let mut chunks = vec![vec![]];
    let mut current_numer_degree = 0;
    let mut current_denom_degree = 0;
    interaction_degrees
        .into_iter()
        .for_each(|(idx, field_degree, count_degree)| {
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
        });
    chunks
}
