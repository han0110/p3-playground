use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Reverse;
use core::iter::{Take, repeat, repeat_with};
use core::marker::PhantomData;
use core::mem;
use core::ops::{Deref, Range};

use itertools::{Itertools, chain, cloned, enumerate, izip, rev};
use p3_challenger::FieldChallenger;
use p3_commit::{
    ExtensionMmcs, Mmcs, OpenedValues, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField, scale_slice_in_place};
use p3_matrix::dense::{DenseMatrix, DenseStorage, RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::extension::FlatMatrixView;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use whir_p3::fiat_shamir::domain_separator::DomainSeparator;
use whir_p3::fiat_shamir::errors::ProofError;
use whir_p3::fiat_shamir::pow::blake3::Blake3PoW;
use whir_p3::parameters::{MultivariateParameters, WhirParameters};
use whir_p3::poly::coeffs::CoefficientList;
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;
use whir_p3::whir::committer::Witness;
use whir_p3::whir::committer::reader::ParsedCommitment;
use whir_p3::whir::domainsep::WhirDomainSeparator;
use whir_p3::whir::parameters::WhirConfig;
use whir_p3::whir::prover::Prover;
use whir_p3::whir::statement::{Statement, StatementVerifier, Weights};
use whir_p3::whir::verifier::Verifier;

#[derive(Debug)]
pub struct UniWhirPcs<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize> {
    dft: Dft,
    whir: WhirParameters<Hash, Compression>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize>
    UniWhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
{
    pub const fn new(dft: Dft, whir: WhirParameters<Hash, Compression>) -> Self {
        Self {
            dft,
            whir,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Challenge: Serialize, [u8; DIGEST_ELEMS]: Serialize",
    deserialize = "Challenge: DeserializeOwned, [u8; DIGEST_ELEMS]: DeserializeOwned"
))]
pub struct WhirProof<Challenge, const DIGEST_ELEMS: usize> {
    pub ood_answers: Vec<Challenge>,
    pub narg_string: Vec<u8>,
    pub proof: whir_p3::whir::WhirProof<Challenge, DIGEST_ELEMS>,
}

impl<Val, Dft, Hash, Compression, Challenge, Challenger, const DIGEST_ELEMS: usize>
    Pcs<Challenge, Challenger> for UniWhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val>,
    Hash: CryptographicHasher<Val, [u8; DIGEST_ELEMS]> + Sync,
    Compression: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
    Challenge: ExtensionField<Val> + TwoAdicField<PrimeSubfield = Val>,
    Challenger: FieldChallenger<Val>,
    [u8; DIGEST_ELEMS]: Serialize + DeserializeOwned,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment =
        <MerkleTreeMmcs<Val, u8, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::Commitment;
    type ProverData = (
        ConcatMats<Val>,
        RefCell<
            Option<
                <MerkleTreeMmcs<Val, u8, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::ProverData<
                    FlatMatrixView<Val, Challenge, RowMajorMatrix<Challenge>>,
                >,
            >,
        >,
    );
    type EvaluationsOnDomain<'a> = Dft::Evaluations;
    type Proof = Vec<WhirProof<Challenge, DIGEST_ELEMS>>;
    type Error = ProofError;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree);
        TwoAdicMultiplicativeCoset {
            log_n,
            shift: Val::ONE,
        }
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let coeffs = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let mut coeffs = self.dft.idft_batch(evals);
                if domain.shift != Val::ONE {
                    let shift_inv = domain.shift.inverse();
                    let powers = shift_inv.powers().take(coeffs.height()).collect_vec();
                    coeffs
                        .par_rows_mut()
                        .zip(powers)
                        .for_each(|(row, power)| scale_slice_in_place(power, row));
                }
                coeffs
            })
            .collect_vec();
        let concat_mats = ConcatMats::new(coeffs);
        let (commitment, merkle_tree) = {
            let width = 1 << self.whir.folding_factor.at_round(0);
            let mut coeffs = evals_to_coeffs(concat_mats.values.clone());
            coeffs.resize(coeffs.len() << self.whir.starting_log_inv_rate, Val::ZERO);
            let folded_coeffs = RowMajorMatrix::new(coeffs, width);
            let folded_codeword = self.dft.dft_batch(folded_coeffs).to_row_major_matrix();
            // TODO(whir-p3): Commit to base elements
            let folded_codeword = RowMajorMatrix::new(
                folded_codeword
                    .values
                    .into_par_iter()
                    .map(Challenge::from)
                    .collect(),
                width,
            );
            let mmcs = ExtensionMmcs::new(MerkleTreeMmcs::new(
                self.whir.merkle_hash.clone(),
                self.whir.merkle_compress.clone(),
            ));
            mmcs.commit(vec![folded_codeword])
        };
        (commitment, (concat_mats, RefCell::new(Some(merkle_tree))))
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        (concat_mats, _): &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let coeffs = {
            let mat = concat_mats.mat(idx);
            let mut coeffs =
                RowMajorMatrix::new(Val::zero_vec(mat.width() << domain.log_n), mat.width());
            coeffs
                .par_rows_mut()
                .zip(mat.par_row_slices())
                .for_each(|(dst, src)| dst.copy_from_slice(src));
            coeffs
        };
        self.dft.coset_dft_batch(coeffs, domain.shift)
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        let opened_values = rounds
            .iter()
            .map(|((concat_mats, _), points)| {
                concat_mats
                    .mats()
                    .zip(points)
                    .map(|(mat, points)| {
                        points
                            .iter()
                            .map(|point| {
                                let powers = point.powers().take(mat.height()).collect_vec();
                                let ys = mat.columnwise_dot_product(&powers);
                                ys.iter()
                                    .for_each(|&y| challenger.observe_algebra_element(y));
                                ys
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        let proofs = izip!(rounds, &opened_values)
            .map(|(((concat_mats, merkle_tree), points), evals)| {
                let config = WhirConfig::<Challenge, Val, Hash, Compression, Blake3PoW>::new(
                    MultivariateParameters::new(concat_mats.meta.num_vars),
                    self.whir.clone(),
                );
                let polynomial = CoefficientList::new(evals_to_coeffs(concat_mats.values.clone()));
                let (ood_points, ood_answers) = repeat_with(|| {
                    let ood_point: Challenge = challenger.sample_algebra_element();
                    let ood_answer = polynomial.evaluate_at_extension(
                        &MultilinearPoint::expand_from_univariate(
                            ood_point,
                            concat_mats.meta.num_vars,
                        ),
                    );
                    (ood_point, ood_answer)
                })
                .take(config.committment_ood_samples)
                .collect::<(Vec<_>, Vec<_>)>();

                let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                    .take(concat_mats.meta.max_log_width())
                    .collect_vec();

                let mut statement = Statement::new(concat_mats.meta.num_vars);
                izip!(&points, evals)
                    .enumerate()
                    .for_each(|(idx, (points, evals))| {
                        izip!(points, evals).for_each(|(point, evals)| {
                            let (weight, eval) =
                                concat_mats.meta.constraint(idx, *point, evals, &r);
                            statement.add_constraint(weight, eval);
                        })
                    });

                // TODO(whir-p3): Use base elements in opening
                let polynomial = CoefficientList::new(
                    polynomial
                        .coeffs()
                        .into_par_iter()
                        .copied()
                        .map(Challenge::from)
                        .collect(),
                );
                let witness = Witness {
                    polynomial,
                    prover_data: merkle_tree.take().unwrap(),
                    ood_points,
                    ood_answers: ood_answers.clone(),
                };
                let mut prover_state = DomainSeparator::new("🌪️")
                    .add_whir_proof(&config)
                    .to_prover_state();
                let proof = Prover(config.clone())
                    .prove(&mut prover_state, statement, witness)
                    .unwrap();
                WhirProof {
                    ood_answers,
                    narg_string: prover_state.narg_string().to_vec(),
                    proof,
                }
            })
            .collect();

        (opened_values, proofs)
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proofs: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        rounds.iter().for_each(|(_, round)| {
            round.iter().for_each(|(_, mat)| {
                mat.iter().for_each(|(_, ys)| {
                    ys.iter()
                        .for_each(|&y| challenger.observe_algebra_element(y))
                })
            })
        });

        izip!(rounds, proofs).try_for_each(|((commitment, mats), proof)| {
            let concat_mats_meta = ConcatMatsMeta::new(
                mats.iter()
                    .map(|(domain, evals)| Dimensions {
                        width: evals[0].1.len(),
                        height: 1 << domain.log_n,
                    })
                    .collect(),
            );

            let config = WhirConfig::<Challenge, Val, Hash, Compression, Blake3PoW>::new(
                MultivariateParameters::new(concat_mats_meta.num_vars),
                self.whir.clone(),
            );

            let ood_points = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                .take(config.committment_ood_samples)
                .collect_vec();

            let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                .take(concat_mats_meta.max_log_width())
                .collect_vec();

            let mut statement = Statement::new(concat_mats_meta.num_vars);
            mats.iter().enumerate().for_each(|(idx, (_, evals))| {
                evals.iter().for_each(|(point, evals)| {
                    let (weight, eval) = concat_mats_meta.constraint(idx, *point, evals, &r);
                    statement.add_constraint(weight, eval);
                })
            });

            let mut verifier_state = DomainSeparator::new("🌪️")
                .add_whir_proof(&config)
                .to_verifier_state(&proof.narg_string);
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
    num_vars: usize,
    dimensions: Vec<Dimensions>,
    ranges: Vec<Range<usize>>,
}

impl ConcatMatsMeta {
    fn new(dims: Vec<Dimensions>) -> Self {
        let (dimensions, ranges) = dims
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dim)| Reverse(dim.width * dim.height))
            .scan(0, |offset, (idx, dim)| {
                let size = dim.width.next_power_of_two() * dim.height;
                let offset = mem::replace(offset, *offset + size);
                Some((idx, dim, offset..offset + size))
            })
            .sorted_by_key(|(idx, _, _)| *idx)
            .map(|(_, dim, range)| (dim, range))
            .collect::<(Vec<_>, Vec<_>)>();
        let num_vars = log2_ceil_usize(
            ranges
                .iter()
                .map(|range| range.end)
                .max()
                .unwrap_or_default(),
        );
        Self {
            num_vars,
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
        x: Challenge,
        ys: &[Challenge],
        r: &[Challenge],
    ) -> (Weights<Challenge>, Challenge) {
        let log_width = log2_ceil_usize(self.dimensions[idx].width);
        let log_height = log2_strict_usize(self.dimensions[idx].height);

        let mut weight = Challenge::zero_vec(1 << self.num_vars);
        weight[self.ranges[idx].start] = Challenge::ONE;
        expand_eq(
            &mut weight[self.ranges[idx].start..][..1 << log_width],
            &r[..log_width],
        );
        expand_pow(
            &mut weight[self.ranges[idx].start..][..1 << (log_width + log_height)],
            x,
            log_height,
        );
        let eval = EvaluationsList::new(
            chain![cloned(ys), repeat(Challenge::ZERO)]
                .take(1 << log_width)
                .collect(),
        )
        .evaluate(&MultilinearPoint(rev(&r[..log_width]).copied().collect()));

        (Weights::linear(EvaluationsList::new(weight)), eval)
    }
}

pub struct ConcatMats<Val> {
    values: Vec<Val>,
    meta: ConcatMatsMeta,
}

impl<Val: Field> ConcatMats<Val> {
    fn new(mats: Vec<RowMajorMatrix<Val>>) -> Self {
        let meta = ConcatMatsMeta::new(mats.iter().map(Matrix::dimensions).collect());
        let mut values = Val::zero_vec(1 << meta.num_vars);
        izip!(&meta.ranges, mats).for_each(|(range, mat)| {
            values[range.clone()]
                .par_chunks_mut(mat.width().next_power_of_two())
                .zip(mat.par_row_slices())
                .for_each(|(dst, src)| dst[..src.len()].copy_from_slice(src));
        });
        Self { values, meta }
    }

    fn mats(&self) -> impl Iterator<Item = RstripedMatrixView<Val, RowMajorMatrixView<Val>>> {
        izip!(&self.meta.dimensions, &self.meta.ranges).map(|(dim, range)| {
            RstripedMatrixView::new(
                RowMajorMatrixView::new(&self.values[range.clone()], dim.width.next_power_of_two()),
                dim.width,
            )
        })
    }

    fn mat(&self, idx: usize) -> RstripedMatrixView<Val, RowMajorMatrixView<Val>> {
        RstripedMatrixView::new(
            RowMajorMatrixView::new(
                &self.values[self.meta.ranges[idx].clone()],
                self.meta.dimensions[idx].width.next_power_of_two(),
            ),
            self.meta.dimensions[idx].width,
        )
    }
}

pub struct RstripedMatrixRowSlice<R> {
    inner: R,
    end: usize,
}

impl<R> RstripedMatrixRowSlice<R> {
    #[inline]
    pub const fn new(inner: R, end: usize) -> Self {
        Self { inner, end }
    }
}

impl<T, R: Deref<Target = [T]>> Deref for RstripedMatrixRowSlice<R> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner[..self.end]
    }
}

pub struct RstripedMatrixView<T, M> {
    inner: M,
    width: usize,
    _marker: PhantomData<T>,
}

impl<T, M> RstripedMatrixView<T, M> {
    #[inline]
    pub const fn new(inner: M, width: usize) -> Self {
        Self {
            inner,
            width,
            _marker: PhantomData,
        }
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> RstripedMatrixView<T, DenseMatrix<T, S>> {
    pub fn par_row_slices(&self) -> impl IndexedParallelIterator<Item = &[T]>
    where
        T: Sync,
    {
        self.inner
            .values
            .borrow()
            .par_chunks_exact(self.inner.width)
            .map(|chunk| &chunk[..self.width])
    }
}

impl<M: Matrix<T>, T: Send + Sync> Matrix<T> for RstripedMatrixView<T, M> {
    type Row<'a>
        = Take<M::Row<'a>>
    where
        Self: 'a;

    #[inline]
    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(r).take(self.width)
    }

    #[inline]
    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        RstripedMatrixRowSlice::new(self.inner.row_slice(r), self.width)
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.height()
    }
}

fn expand_eq<Challenge: Field>(evals: &mut [Challenge], r: &[Challenge]) {
    let size = evals.len() >> r.len();
    enumerate(r).for_each(|(i, r_i)| {
        let (lo, hi) = evals.split_at_mut(size << i);
        lo.par_iter_mut().zip(hi).for_each(|(lo, hi)| {
            *hi = *lo * *r_i;
            *lo -= *hi;
        });
    });
}

fn expand_pow<Challenge: Field>(evals: &mut [Challenge], x: Challenge, n: usize) {
    let size = evals.len() >> n;
    let mut x_sqr = x;
    (0..n).for_each(|i| {
        let (lo, hi) = evals.split_at_mut(size << i);
        lo.par_iter().zip(hi).for_each(|(lo, hi)| *hi = *lo * x_sqr);
        x_sqr = x_sqr.square();
    });
}

fn evals_to_coeffs<F: Field>(mut evals: Vec<F>) -> Vec<F> {
    (0..log2_strict_usize(evals.len())).for_each(|i| {
        evals.par_chunks_mut(2 << i).for_each(|chunk| {
            let (lo, hi) = chunk.split_at_mut(1 << i);
            izip!(lo, hi).for_each(|(lo, hi)| *hi -= *lo);
        });
    });
    evals
}
