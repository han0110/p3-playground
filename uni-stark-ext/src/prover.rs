use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use core::ops::Deref;

use itertools::{Itertools, chain, izip};
use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{
    BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing, batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{debug_span, info_span, instrument};

use crate::{
    Commitments, Domain, OpenedValues, PackedChallenge, PackedVal, Proof, ProofPerAir,
    ProverConstraintFolder, ProverInteractionFolder, StarkGenericConfig, SymbolicAirBuilder, Val,
    VerifierInput, eval_log_up,
};

#[derive(Clone, Debug)]
pub struct ProverInput<Val, A> {
    pub(crate) inner: VerifierInput<Val, A>,
    pub(crate) trace: RowMajorMatrix<Val>,
}

impl<Val, A> Deref for ProverInput<Val, A> {
    type Target = VerifierInput<Val, A>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Val: Field, A> ProverInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>, trace: RowMajorMatrix<Val>) -> Self
    where
        A: Air<SymbolicAirBuilder<Val>>,
    {
        Self {
            inner: VerifierInput::new(air, public_values),
            trace,
        }
    }
}

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove<
    SC,
    #[cfg(feature = "check-constraints")] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(feature = "check-constraints"))] A,
>(
    config: &SC,
    inputs: Vec<ProverInput<Val<SC>, A>>,
    challenger: &mut SC::Challenger,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>>
        + for<'a> Air<ProverInteractionFolder<'a, SC>>
        + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    #[cfg(feature = "check-constraints")]
    crate::check_constraints::check_constraints(&inputs);

    let (inputs, main_traces, log_degrees) = inputs
        .into_iter()
        .map(|input| {
            let log_degree = log2_strict_usize(input.trace.height());
            (input.inner, input.trace, log_degree)
        })
        .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

    let has_any_interaction = inputs.iter().any(VerifierInput::has_interaction);

    let pcs = config.pcs();
    let (main_domains, quotient_domains) = izip!(&inputs, &log_degrees)
        .map(|(input, log_degree)| {
            let main_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let quotient_domain =
                main_domain.create_disjoint_domain(1 << (log_degree + input.log_quotient_degree));
            (main_domain, quotient_domain)
        })
        .collect::<(Vec<_>, Vec<_>)>();

    let (main_commit, main_data) = info_span!("commit to main data").in_scope(|| {
        pcs.commit(izip!(main_domains.iter().copied(), main_traces.clone()).collect())
    });

    // Observe the instance.
    // degree < 2^255 so we can safely cast log_degree to a u8.
    log_degrees
        .iter()
        .for_each(|log_degree| challenger.observe(Val::<SC>::from_u8(*log_degree as u8)));
    // TODO: Might be best practice to include other instance data here; see verifier comment.

    challenger.observe(main_commit.clone());
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(&input.public_values));

    let beta: SC::Challenge = challenger.sample_algebra_element();
    let gamma: SC::Challenge = challenger.sample_algebra_element();
    let max_bus_index =
        itertools::max(inputs.iter().map(|input| input.max_bus_index)).unwrap_or_default();
    let max_field_count =
        itertools::max(inputs.iter().map(|input| input.max_field_count)).unwrap_or_default();
    let beta_powers = beta.powers().skip(1).take(max_field_count).collect_vec();
    let gamma_powers = gamma.powers().skip(1).take(max_bus_index + 1).collect_vec();

    let (log_up_sums, log_up_traces) = izip!(&inputs, main_traces)
        .map(|(input, main_trace)| {
            (input.has_interaction())
                .then(|| {
                    log_up_trace(
                        &input.air,
                        &input.public_values,
                        input.interaction_count,
                        &input.interaction_chunks,
                        main_trace.as_view(),
                        &beta_powers,
                        &gamma_powers,
                    )
                })
                .unzip()
        })
        .collect::<(Vec<_>, Vec<_>)>();

    #[cfg(feature = "check-constraints")]
    assert_eq!(
        log_up_sums.iter().flatten().copied().sum::<SC::Challenge>(),
        SC::Challenge::ZERO
    );

    log_up_sums
        .iter()
        .flatten()
        .for_each(|sum| challenger.observe_algebra_element(*sum));

    let (log_up_commit, log_up_data) = info_span!("commit to log up data").in_scope(|| {
        has_any_interaction
            .then(|| {
                pcs.commit(
                    izip!(&main_domains, log_up_traces)
                        .flat_map(|(main_domain, trace)| {
                            trace.map(|trace| (*main_domain, trace.flatten_to_base()))
                        })
                        .collect(),
                )
            })
            .unzip()
    });

    if let Some(log_up_commit) = &log_up_commit {
        challenger.observe(log_up_commit.clone());
    }

    let alpha: SC::Challenge = challenger.sample_algebra_element();

    let max_constraint_count =
        itertools::max(inputs.iter().map(|input| input.constraint_count)).unwrap_or_default();
    let mut alpha_powers = alpha.powers().take(max_constraint_count).collect_vec();
    alpha_powers.reverse();
    let beta_powers = beta_powers.into_iter().map_into().collect_vec();
    let gamma_powers = gamma_powers.into_iter().map_into().collect_vec();

    let quotient_values = izip!(&inputs, &main_domains, &quotient_domains, &log_up_sums)
        .enumerate()
        .map(|(idx, (input, main_domain, quotient_domain, log_up_sum))| {
            let main_trace_on_quotient_domain =
                pcs.get_evaluations_on_domain(&main_data, idx, *quotient_domain);
            let log_up_trace_on_quotient_domain = log_up_data.as_ref().and_then(|log_up_data| {
                (input.has_interaction())
                    .then(|| pcs.get_evaluations_on_domain(log_up_data, idx, *quotient_domain))
            });
            quotient_values(
                &input.air,
                &input.public_values,
                input.interaction_count,
                &input.interaction_chunks,
                *main_domain,
                *quotient_domain,
                main_trace_on_quotient_domain,
                log_up_trace_on_quotient_domain,
                &alpha_powers[max_constraint_count - input.constraint_count..],
                &beta_powers,
                &gamma_powers,
                log_up_sum.unwrap_or_default().into(),
            )
        })
        .collect::<Vec<_>>();

    let quotient_chunks = izip!(&inputs, &quotient_domains, quotient_values)
        .flat_map(|(input, quotient_domain, quotient_values)| {
            let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
            let quotient_chunks =
                quotient_domain.split_evals(input.quotient_degree(), quotient_flat);
            let qc_domains = quotient_domain.split_domains(input.quotient_degree());
            izip!(qc_domains, quotient_chunks)
        })
        .collect();

    let (quotient_commit, quotient_data) =
        info_span!("commit to quotient poly chunks").in_scope(|| pcs.commit(quotient_chunks));
    challenger.observe(quotient_commit.clone());

    let commitments = Commitments {
        main: main_commit,
        log_up_chunks: log_up_commit,
        quotient_chunks: quotient_commit,
    };

    let zeta: SC::Challenge = challenger.sample();

    let (opened_values, opening_proof) = info_span!("open").in_scope(|| {
        pcs.open(
            chain![
                Some((
                    &main_data,
                    log_degrees
                        .iter()
                        .map(|log_degree| {
                            let main_domain = pcs.natural_domain_for_degree(1 << log_degree);
                            let zeta_next = main_domain.next_point(zeta).unwrap();
                            vec![zeta, zeta_next]
                        })
                        .collect(),
                )),
                log_up_data.as_ref().map(|log_up_data| (
                    log_up_data,
                    izip!(&inputs, &log_degrees)
                        .filter(|(input, _)| input.has_interaction())
                        .map(|(_, log_degree)| {
                            let main_domain = pcs.natural_domain_for_degree(1 << log_degree);
                            let zeta_next = main_domain.next_point(zeta).unwrap();
                            vec![zeta, zeta_next]
                        })
                        .collect(),
                )),
                Some((
                    &quotient_data,
                    // open every chunk at zeta
                    vec![vec![zeta]; inputs.iter().map(VerifierInput::quotient_degree).sum()],
                )),
            ]
            .collect(),
            challenger,
        )
    });
    let mut opened_values = opened_values.into_iter();
    let main = opened_values.next().unwrap();
    let mut log_up_chunks = has_any_interaction.then(|| opened_values.next().unwrap().into_iter());
    let mut quotient_chunks = opened_values.next().unwrap().into_iter();
    let per_air = izip!(&inputs, log_degrees, log_up_sums, main)
        .map(|(input, log_degree, log_up_sum, main)| {
            let (main_local, main_next) = {
                let mut main = main.into_iter();
                (main.next().unwrap(), main.next().unwrap())
            };
            let (log_up_local, log_up_next) = log_up_chunks
                .as_mut()
                .and_then(|values| {
                    input.has_interaction().then(|| {
                        let mut values = values.next().unwrap().into_iter();
                        (values.next().unwrap(), values.next().unwrap())
                    })
                })
                .unwrap_or_default();
            let quotient_chunks = quotient_chunks
                .by_ref()
                .map(|mut values| values.pop().unwrap())
                .take(input.quotient_degree())
                .collect_vec();
            let opened_values = OpenedValues {
                main_local,
                main_next,
                log_up_local,
                log_up_next,
                quotient_chunks,
            };
            ProofPerAir {
                log_degree,
                log_up_sum,
                opened_values,
            }
        })
        .collect_vec();

    Proof {
        commitments,
        per_air,
        opening_proof,
    }
}

#[instrument(name = "compute log up trace", skip_all)]
fn log_up_trace<SC, A>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    interaction_count: usize,
    interaction_chunks: &[Vec<usize>],
    main_trace: RowMajorMatrixView<Val<SC>>,
    beta_powers: &[SC::Challenge],
    gamma_powers: &[SC::Challenge],
) -> (SC::Challenge, RowMajorMatrix<SC::Challenge>)
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverInteractionFolder<'a, SC>>,
{
    let width = interaction_chunks.len() + 1;
    let height = main_trace.height();
    let mut trace = RowMajorMatrix::new(SC::Challenge::zero_vec(width * height), width);

    let num_threads = if cfg!(feature = "parallel") {
        p3_maybe_rayon::prelude::current_num_threads()
    } else {
        1
    };
    let chunk_rows = height.div_ceil(num_threads);

    trace
        .par_row_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(i, mut trace_chunk)| {
            let mut numers = Val::<SC>::zero_vec(interaction_count * trace_chunk.height());
            let mut denoms = SC::Challenge::zero_vec(interaction_count * trace_chunk.height());
            numers
                .par_chunks_mut(interaction_count)
                .zip(denoms.par_chunks_mut(interaction_count))
                .enumerate()
                .for_each(|(j, (numers, denoms))| {
                    let row_idx = i * chunk_rows + j;

                    let local = main_trace.row_slice(row_idx);
                    let next = main_trace.row_slice((row_idx + 1) % height);
                    let main = VerticalPair::new(
                        RowMajorMatrixView::new_row(&local),
                        RowMajorMatrixView::new_row(&next),
                    );
                    let mut builder = ProverInteractionFolder {
                        main,
                        public_values,
                        beta_powers,
                        gamma_powers,
                        numers,
                        denoms,
                        interaction_index: 0,
                    };
                    air.eval(&mut builder);
                });

            let denom_invs = batch_multiplicative_inverse(&denoms);
            drop(denoms);

            trace_chunk
                .par_rows_mut()
                .zip(
                    numers
                        .par_chunks(interaction_count)
                        .zip(denom_invs.par_chunks(interaction_count)),
                )
                .for_each(|(row, (numers, denom_invs))| {
                    let mut sum = SC::Challenge::ZERO;
                    izip!(interaction_chunks, &mut *row).for_each(|(chunk, value)| {
                        chunk
                            .iter()
                            .for_each(|idx| *value += denom_invs[*idx] * numers[*idx]);
                        sum += *value;
                    });
                    row[width - 1] = sum;
                });
        });

    let mut sum = SC::Challenge::ZERO;
    trace.values.chunks_mut(width).for_each(|row| {
        sum += row[width - 1];
        row[width - 1] = sum;
    });

    (sum, trace)
}

#[instrument(name = "compute quotient polynomial", skip_all)]
#[allow(clippy::too_many_arguments)]
fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    interaction_count: usize,
    interaction_chunks: &[Vec<usize>],
    main_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    main_trace_on_quotient_domain: Mat,
    log_up_trace_on_quotient_domain: Option<Mat>,
    alpha_powers: &[SC::Challenge],
    beta_powers: &[PackedChallenge<SC>],
    gamma_powers: &[PackedChallenge<SC>],
    log_up_sum: PackedChallenge<SC>,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = main_trace_on_quotient_domain.width();
    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| main_domain.selectors_on_coset(quotient_domain));

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(main_domain.size());
    let next_step = 1 << qdb;

    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }

    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

            let main = RowMajorMatrix::new(
                main_trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                width,
            );

            let accumulator = PackedChallenge::<SC>::ZERO;
            let mut folder = ProverConstraintFolder {
                main: main.as_view(),
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha_powers,
                accumulator,
                constraint_index: 0,
                beta_powers,
                gamma_powers,
                numers: vec![Default::default(); interaction_count],
                denoms: vec![Default::default(); interaction_count],
                interaction_index: 0,
            };
            air.eval(&mut folder);

            if let Some(log_up_trace_on_quotient_domain) = &log_up_trace_on_quotient_domain {
                let numers = mem::take(&mut folder.numers);
                let denoms = mem::take(&mut folder.denoms);
                let [log_up_local, log_up_next] = [i_start, i_start + next_step].map(|row| {
                    (0..log_up_trace_on_quotient_domain.width())
                        .step_by(SC::Challenge::DIMENSION)
                        .map(|col| {
                            PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                                PackedVal::<SC>::from_fn(|j| {
                                    log_up_trace_on_quotient_domain
                                        .get((row + j) % quotient_size, col + i)
                                })
                            })
                        })
                        .collect_vec()
                });
                eval_log_up(
                    &mut folder,
                    &numers,
                    &denoms,
                    &log_up_local,
                    &log_up_next,
                    interaction_chunks,
                    log_up_sum,
                );
            }

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.accumulator * inv_vanishing;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
                    quotient.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}
