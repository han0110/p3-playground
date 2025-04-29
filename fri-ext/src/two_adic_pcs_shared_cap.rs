use alloc::collections::btree_map::BTreeMap;
use alloc::collections::btree_set::BTreeSet;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{BatchOpening, FriConfig, FriProof, TwoAdicFriPcs};

pub struct TwoAdicFriPcsSharedCap<Val, Dft, InputMmcs, FriMmcs, Digest> {
    inner: TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>,
    shared_cap: usize,
    _marker: PhantomData<Digest>,
}

impl<Val, Dft, InputMmcs, FriMmcs, Digest>
    TwoAdicFriPcsSharedCap<Val, Dft, InputMmcs, FriMmcs, Digest>
{
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        let shared_cap = fri.num_queries.ilog2() as usize;
        Self {
            inner: TwoAdicFriPcs::new(dft, mmcs, fri),
            shared_cap,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize, Digest: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>, Digest: Deserialize<'de>"
))]
pub struct FriProofSharedCap<F: Field, M: Mmcs<F>, Witness, InputProof, Digest> {
    pub inner: FriProof<F, M, Witness, InputProof>,
    pub input_shared_digest_set: Vec<Digest>,
    pub input_shared_digest_indices: Vec<Vec<Vec<u16>>>,
    pub commit_phase_shared_digest_set: Vec<Digest>,
    pub commit_phase_shared_digest_indices: Vec<Vec<Vec<u16>>>,
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger, Digest> Pcs<Challenge, Challenger>
    for TwoAdicFriPcsSharedCap<Val, Dft, InputMmcs, FriMmcs, Digest>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, Proof = Vec<Digest>>,
    FriMmcs: Mmcs<Challenge, Proof = Vec<Digest>>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    Digest: Clone + Ord + Send + Sync + Serialize + DeserializeOwned,
{
    type Domain =
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::Domain;
    type Commitment =
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::Commitment;
    type ProverData =
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::ProverData;
    type EvaluationsOnDomain<'a> = <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<
        Challenge,
        Challenger,
    >>::EvaluationsOnDomain<'a>;
    type Proof =
        FriProofSharedCap<Challenge, FriMmcs, Val, Vec<BatchOpening<Val, InputMmcs>>, Digest>;
    type Error = <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::Error;

    const ZK: bool =
        <TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> as Pcs<Challenge, Challenger>>::ZK;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        Pcs::<Challenge, Challenger>::natural_domain_for_degree(&self.inner, degree)
    }

    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        Pcs::<Challenge, Challenger>::commit(&self.inner, evaluations)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        Pcs::<Challenge, Challenger>::get_evaluations_on_domain(
            &self.inner,
            prover_data,
            idx,
            domain,
        )
    }

    fn open(
        &self,
        rounds: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        let (opened_values, mut proof) =
            Pcs::<Challenge, Challenger>::open(&self.inner, rounds, challenger);

        let shared_cap = self.shared_cap;
        let (input_shared_digests, commit_phase_shared_digests) = proof
            .query_proofs
            .par_iter_mut()
            .map(|query_proof| {
                (
                    query_proof
                        .input_proof
                        .par_iter_mut()
                        .map(|opening| {
                            let start = opening.opening_proof.len().saturating_sub(shared_cap);
                            opening.opening_proof.drain(start..).collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                    query_proof
                        .commit_phase_openings
                        .par_iter_mut()
                        .map(|opening| {
                            let start = opening.opening_proof.len().saturating_sub(shared_cap);
                            opening.opening_proof.drain(start..).collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<(Vec<_>, Vec<_>)>();
        let index = |shared_digests: Vec<Vec<Vec<_>>>| {
            let shared_digest_set = Vec::from_iter(
                shared_digests
                    .par_iter()
                    .flatten()
                    .flatten()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .cloned(),
            );
            let shared_digest_indices = {
                let map = shared_digest_set
                    .par_iter()
                    .enumerate()
                    .map(|(v, k)| (k, v as u16))
                    .collect::<BTreeMap<_, _>>();
                shared_digests
                    .into_par_iter()
                    .map(|v| {
                        v.into_par_iter()
                            .map(|v| v.into_par_iter().map(|v| map[&&v]).collect::<Vec<_>>())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            (shared_digest_set, shared_digest_indices)
        };
        let (
            (input_shared_digest_set, input_shared_digest_indices),
            (commit_phase_shared_digest_set, commit_phase_shared_digest_indices),
        ) = join(
            || index(input_shared_digests),
            || index(commit_phase_shared_digests),
        );

        (
            opened_values,
            FriProofSharedCap {
                inner: proof,
                input_shared_digest_set,
                input_shared_digest_indices,
                commit_phase_shared_digest_set,
                commit_phase_shared_digest_indices,
            },
        )
    }

    #[allow(clippy::type_complexity)]
    fn verify(
        &self,
        rounds: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Challenge, Vec<Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let FriProofSharedCap {
            inner: proof,
            input_shared_digest_set,
            input_shared_digest_indices,
            commit_phase_shared_digest_set,
            commit_phase_shared_digest_indices,
        } = proof;
        let mut proof = proof.clone();

        // TODO: Check indices have expected length and all index is in valid range.

        proof
            .query_proofs
            .par_iter_mut()
            .zip(input_shared_digest_indices)
            .zip(commit_phase_shared_digest_indices)
            .for_each(|((query_proof, input_indices), commit_phase_indices)| {
                query_proof
                    .input_proof
                    .par_iter_mut()
                    .zip(input_indices)
                    .for_each(|(opening, indices)| {
                        opening.opening_proof.extend(
                            indices
                                .iter()
                                .map(|idx| input_shared_digest_set[*idx as usize].clone()),
                        );
                    });
                query_proof
                    .commit_phase_openings
                    .par_iter_mut()
                    .zip(commit_phase_indices)
                    .for_each(|(opening, indices)| {
                        opening.opening_proof.extend(
                            indices
                                .iter()
                                .map(|idx| commit_phase_shared_digest_set[*idx as usize].clone()),
                        );
                    });
            });

        Pcs::<Challenge, Challenger>::verify(&self.inner, rounds, &proof, challenger)
    }
}
