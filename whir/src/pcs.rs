use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Reverse;
use core::iter::repeat_with;
use core::marker::PhantomData;
use core::mem::replace;
use core::ops::Range;

use itertools::{Itertools, chain, cloned, izip, rev};
use p3_challenger::FieldChallenger;
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField, dot_product};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::horizontally_truncated::HorizontallyTruncated;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_ml_pcs::{MlPcs, MlQuery, eq_poly};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::info_span;
use whir_p3::fiat_shamir::domain_separator::DomainSeparator;
use whir_p3::fiat_shamir::errors::ProofError;
use whir_p3::fiat_shamir::pow::blake3::Blake3PoW;
use whir_p3::parameters::{MultivariateParameters, ProtocolParameters};
use whir_p3::poly::coeffs::CoefficientList;
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;
use whir_p3::poly::wavelet::inverse_wavelet_transform;
use whir_p3::whir::committer::Witness;
use whir_p3::whir::committer::reader::ParsedCommitment;
use whir_p3::whir::parameters::WhirConfig;
use whir_p3::whir::prover::Prover;
use whir_p3::whir::statement::{Statement, StatementVerifier, Weights};
use whir_p3::whir::verifier::Verifier;

#[derive(Debug)]
pub struct WhirPcs<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize> {
    dft: Dft,
    whir: ProtocolParameters<Hash, Compression>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize>
    WhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
{
    pub const fn new(dft: Dft, whir: ProtocolParameters<Hash, Compression>) -> Self {
        Self {
            dft,
            whir,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Val: Serialize, Challenge: Serialize, [u8; DIGEST_ELEMS]: Serialize",
    deserialize = "Val: DeserializeOwned, Challenge: DeserializeOwned, [u8; DIGEST_ELEMS]: DeserializeOwned"
))]
pub struct WhirProof<Val, Challenge, const DIGEST_ELEMS: usize> {
    pub ood_answers: Vec<Challenge>,
    pub narg_string: Vec<u8>,
    pub proof: whir_p3::whir::prover::proof::WhirProof<Val, Challenge, DIGEST_ELEMS>,
}

impl<Val, Dft, Hash, Compression, Challenge, Challenger, const DIGEST_ELEMS: usize>
    MlPcs<Challenge, Challenger> for WhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val>,
    Hash: Sync + CryptographicHasher<Val, [u8; DIGEST_ELEMS]>,
    Compression: Sync + PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger: FieldChallenger<Val>,
    [u8; DIGEST_ELEMS]: Serialize + DeserializeOwned,
{
    type Val = Val;
    type Commitment =
        <MerkleTreeMmcs<Val, u8, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::Commitment;
    type ProverData = (
        ConcatMats<Val>,
        // TODO(whir-p3): Use reference to merkle tree in `Witness` to avoid cloning or ownership taking.
        RefCell<
            Option<
                <MerkleTreeMmcs<Val, u8, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::ProverData<
                    RowMajorMatrix<Val>,
                >,
            >,
        >,
    );
    type Evaluations<'a> = HorizontallyTruncated<Val, RowMajorMatrixView<'a, Val>>;
    type Proof = Vec<WhirProof<Val, Challenge, DIGEST_ELEMS>>;
    type Error = ProofError;

    fn commit(
        &self,
        evaluations: Vec<RowMajorMatrix<Self::Val>>,
    ) -> (Self::Commitment, Self::ProverData) {
        // Concat matrices into single polynomial.
        let concat_mats = info_span!("concat matrices").in_scope(|| ConcatMats::new(evaluations));

        // This should generate the same codeword and commitment as in `whir_p3`.
        let (commitment, merkle_tree) = {
            // TODO(whir-p3): Commit to evaluation form directly and fold by `eq([lo, hi], r)`.
            let coeffs = info_span!("evals to coeffs").in_scope(|| {
                let size = 1 << (concat_mats.meta.log_b + self.whir.starting_log_inv_rate);
                let mut coeffs = Vec::with_capacity(size);
                coeffs.extend(&concat_mats.values);
                inverse_wavelet_transform(&mut DenseMatrix::new_col(&mut coeffs));
                coeffs.resize(size, Val::ZERO);
                coeffs
            });
            let folded_codeword = info_span!("compute folded codeword").in_scope(|| {
                let width = 1 << self.whir.folding_factor.at_round(0);
                let folded_coeffs = RowMajorMatrix::new(coeffs, width);
                self.dft.dft_batch(folded_coeffs).to_row_major_matrix()
            });
            let mmcs = MerkleTreeMmcs::new(
                self.whir.merkle_hash.clone(),
                self.whir.merkle_compress.clone(),
            );
            mmcs.commit(vec![folded_codeword])
        };

        (commitment, (concat_mats, RefCell::new(Some(merkle_tree))))
    }

    fn get_evaluations<'a>(
        &self,
        (concat_mats, _): &'a Self::ProverData,
        idx: usize,
    ) -> Self::Evaluations<'a> {
        concat_mats.mat(idx)
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // for each query:
                Vec<(
                    // the query,
                    MlQuery<Challenge>,
                    // values at the query
                    Vec<Challenge>,
                )>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        let proofs = rounds
            .iter()
            .map(|((concat_mats, merkle_tree), queries_and_evals)| {
                let config = WhirConfig::<Challenge, Val, Hash, Compression, Blake3PoW>::new(
                    MultivariateParameters::new(concat_mats.meta.log_b),
                    self.whir.clone(),
                );

                let polynomial = info_span!("evals to coeffs").in_scope(|| {
                    let mut coeffs = concat_mats.values.clone();
                    inverse_wavelet_transform(&mut DenseMatrix::new_col(&mut coeffs));
                    CoefficientList::new(coeffs)
                });
                let (ood_points, ood_answers) = info_span!("compute ood answers").in_scope(|| {
                    repeat_with(|| {
                        let ood_point: Challenge = challenger.sample_algebra_element();
                        let ood_answer = polynomial.evaluate_at_extension(
                            &MultilinearPoint::expand_from_univariate(
                                ood_point,
                                concat_mats.meta.log_b,
                            ),
                        );
                        (ood_point, ood_answer)
                    })
                    .take(config.committment_ood_samples)
                    .collect::<(Vec<_>, Vec<_>)>()
                });

                // Challenge for random linear combining columns.
                let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                    .take(concat_mats.meta.max_log_width())
                    .collect_vec();

                let statement = info_span!("compute weights").in_scope(|| {
                    let mut statement = Statement::new(concat_mats.meta.log_b);
                    queries_and_evals
                        .iter()
                        .enumerate()
                        .for_each(|(idx, queries_and_evals)| {
                            queries_and_evals.iter().for_each(|(query, evals)| {
                                let (weights, sum) =
                                    concat_mats.meta.constraint(idx, query, evals, &r);
                                statement.add_constraint(weights, sum);
                            })
                        });
                    statement
                });

                let mut prover_state = {
                    let mut domainsep = DomainSeparator::new("🌪️");
                    domainsep.add_whir_proof(&config);
                    domainsep.to_prover_state()
                };
                let proof = info_span!("prove").in_scope(|| {
                    let witness = Witness {
                        polynomial,
                        prover_data: merkle_tree.take().unwrap(),
                        ood_points,
                        ood_answers: ood_answers.clone(),
                    };
                    Prover(config)
                        .prove(&self.dft, &mut prover_state, statement, witness)
                        .unwrap()
                });
                WhirProof {
                    ood_answers,
                    narg_string: prover_state.narg_string().to_vec(),
                    proof,
                }
            })
            .collect();

        proofs
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<
                // for each query:
                Vec<(
                    // the query,
                    MlQuery<Challenge>,
                    // values at the query
                    Vec<Challenge>,
                )>,
            >,
        )>,
        proofs: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        izip!(rounds, proofs).try_for_each(|((commitment, round), proof)| {
            let concat_mats_meta = ConcatMatsMeta::new(
                round
                    .iter()
                    .map(|mat| Dimensions {
                        width: mat[0].1.len(),
                        height: 1 << mat[0].0.log_b(),
                    })
                    .collect(),
            );

            let config = WhirConfig::<Challenge, Val, Hash, Compression, Blake3PoW>::new(
                MultivariateParameters::new(concat_mats_meta.log_b),
                self.whir.clone(),
            );

            let ood_points = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                .take(config.committment_ood_samples)
                .collect_vec();

            let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                .take(concat_mats_meta.max_log_width())
                .collect_vec();

            let mut statement = Statement::new(concat_mats_meta.log_b);
            round.iter().enumerate().for_each(|(idx, evals)| {
                evals.iter().for_each(|(query, evals)| {
                    let (weights, sum) = concat_mats_meta.constraint(idx, query, evals, &r);
                    statement.add_constraint(weights, sum);
                })
            });

            let mut verifier_state = {
                let mut domainsep = DomainSeparator::new("🌪️");
                domainsep.add_whir_proof(&config);
                domainsep.to_verifier_state(&proof.narg_string)
            };
            Verifier::new(&config).verify(
                &mut verifier_state,
                &ParsedCommitment {
                    root: commitment,
                    ood_points,
                    ood_answers: proof.ood_answers.clone(),
                },
                &StatementVerifier::from_statement(&statement),
                &proof.proof,
            )
        })?;

        Ok(())
    }
}

pub struct ConcatMatsMeta {
    log_b: usize,
    dimensions: Vec<Dimensions>,
    ranges: Vec<Range<usize>>,
}

impl ConcatMatsMeta {
    fn new(dims: Vec<Dimensions>) -> Self {
        let (dimensions, ranges) = dims
            .iter()
            .enumerate()
            // Sorted by matrix size in descending order.
            .sorted_by_key(|(_, dim)| Reverse(dim.width * dim.height))
            // Calculate sub-cube range for each matrix (power-of-2 aligned).
            .scan(0, |offset, (idx, dim)| {
                let size = dim.width.next_power_of_two() * dim.height;
                let offset = replace(offset, *offset + size);
                Some((idx, dim, offset..offset + size))
            })
            // Store the dimension and range in original order.
            .sorted_by_key(|(idx, _, _)| *idx)
            .map(|(_, dim, range)| (dim, range))
            .collect::<(Vec<_>, Vec<_>)>();
        // Calculate number of variable of concated polynomial.
        let log_b = log2_ceil_usize(
            ranges
                .iter()
                .map(|range| range.end)
                .max()
                .unwrap_or_default(),
        );
        Self {
            log_b,
            dimensions,
            ranges,
        }
    }

    fn max_log_width(&self) -> usize {
        self.dimensions
            .iter()
            .map(|dim| log2_ceil_usize(dim.width))
            .max()
            .unwrap_or_default()
    }

    fn constraint<Challenge: Field>(
        &self,
        idx: usize,
        query: &MlQuery<Challenge>,
        ys: &[Challenge],
        r: &[Challenge],
    ) -> (Weights<Challenge>, Challenge) {
        let log_width = log2_ceil_usize(self.dimensions[idx].width);

        let r = &r[..log_width];
        let eq_r = eq_poly(r, Challenge::ONE);

        let sum = dot_product(cloned(ys), cloned(&eq_r[..ys.len()]));

        let weights = match query {
            MlQuery::Eq(z) => {
                let point = rev(chain![
                    cloned(r),
                    cloned(z),
                    (log2_strict_usize(self.ranges[idx].len())..self.log_b)
                        .map(|i| Challenge::from_bool((self.ranges[idx].start >> i) & 1 == 1))
                ])
                .collect();
                Weights::evaluation(MultilinearPoint(point))
            }
            // TODO(whir-p3): Introduce a new weights variant to generate such evaluations.
            MlQuery::EqRotateRight(_, _) => {
                let mut weight = Challenge::zero_vec(1 << self.log_b);
                weight[self.ranges[idx].clone()]
                    .par_chunks_mut(eq_r.len())
                    .zip(query.to_mle(Challenge::ONE))
                    .for_each(|(weight, query)| {
                        izip!(weight, &eq_r).for_each(|(weight, eq_r)| *weight = *eq_r * query)
                    });
                Weights::linear(EvaluationsList::new(weight))
            }
        };

        (weights, sum)
    }
}

pub struct ConcatMats<Val> {
    values: Vec<Val>,
    meta: ConcatMatsMeta,
}

impl<Val: Field> ConcatMats<Val> {
    fn new(mats: Vec<RowMajorMatrix<Val>>) -> Self {
        let meta = ConcatMatsMeta::new(mats.iter().map(Matrix::dimensions).collect());
        let mut values = Val::zero_vec(1 << meta.log_b);
        izip!(&meta.ranges, mats).for_each(|(range, mat)| {
            // Copy and pad each row into power-of-2 length into concated polynomial.
            values[range.clone()]
                .par_chunks_mut(mat.width().next_power_of_two())
                .zip(mat.par_row_slices())
                .for_each(|(dst, src)| dst[..src.len()].copy_from_slice(src));
        });
        Self { values, meta }
    }

    fn mat(&self, idx: usize) -> HorizontallyTruncated<Val, RowMajorMatrixView<Val>> {
        HorizontallyTruncated::new(
            RowMajorMatrixView::new(
                &self.values[self.meta.ranges[idx].clone()],
                self.meta.dimensions[idx].width.next_power_of_two(),
            ),
            self.meta.dimensions[idx].width,
        )
        .unwrap()
    }
}
