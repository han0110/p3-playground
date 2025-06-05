use alloc::vec::Vec;

use p3_commit::Pcs;
use serde::{Deserialize, Serialize};

use crate::StarkGenericConfig;

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
    pub commitments: Commitments<Com<SC>>,
    pub per_air: Vec<ProofPerAir<SC>>,
    pub opening_proof: PcsProof<SC>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProofPerAir<SC: StarkGenericConfig> {
    pub log_degree: usize,
    pub log_up_sum: Option<SC::Challenge>,
    pub opened_values: OpenedValues<SC::Challenge>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Commitments<Com> {
    pub main: Com,
    pub log_up_chunks: Option<Com>,
    pub quotient_chunks: Com,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    pub main_local: Vec<Challenge>,
    pub main_next: Vec<Challenge>,
    pub log_up_local: Vec<Challenge>,
    pub log_up_next: Vec<Challenge>,
    pub quotient_chunks: Vec<Vec<Challenge>>,
}
