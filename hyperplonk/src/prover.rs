use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use core::mem::transmute;

use itertools::{Itertools, chain, cloned, izip};
use p3_air::Air;
use p3_air_ext::{ProverInput, VerifierInput};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, PackedValue, PrimeCharacteristicRing, TwoAdicField, dot_product};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    AirProof, AirTrace, BatchSumcheckProof, CompressedRoundPoly, EqHelper, FractionalSumProof,
    PiopProof, Proof, ProverConstraintFolderOnExtension, ProverConstraintFolderOnExtensionPacking,
    ProverConstraintFolderOnPacking, ProverInteractionFolderOnExtension,
    ProverInteractionFolderOnPacking, ProvingKey, RSlice, RegularSumcheckProver,
    UnivariateSkipProof, UnivariateSkipProver, eq_poly,
};

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)]
pub fn prove<
    Val,
    Challenge,
    #[cfg(feature = "check-constraints")] A: for<'a> Air<crate::DebugConstraintBuilder<'a, Val>>,
    #[cfg(not(feature = "check-constraints"))] A,
>(
    pk: &ProvingKey,
    inputs: Vec<ProverInput<Val, A>>,
    mut challenger: impl FieldChallenger<Val>,
) -> Proof<Challenge>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverInteractionFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverInteractionFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    #[cfg(feature = "check-constraints")]
    crate::check_constraints(&inputs);

    assert!(!inputs.is_empty());

    let (inputs, traces) = inputs.into_iter().map_into().collect::<(Vec<_>, Vec<_>)>();
    let log_heights = traces
        .iter()
        .map(|mat| log2_strict_usize(mat.height()))
        .collect_vec();

    // TODO: PCS commit and observe.

    cloned(&log_heights).for_each(|log_height| challenger.observe(Val::from_u8(log_height as u8)));
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    let (_zs, piop) = {
        let traces = traces.iter().map(|trace| trace.as_view()).collect();
        prove_piop(pk, &inputs, traces, &mut challenger)
    };

    // TODO: PCS open

    // TODO: Remove the following sanity checks.
    #[cfg(debug_assertions)]
    izip!(pk.metas(), &traces, &piop.air.univariate_skips, &_zs)
        .enumerate()
        .for_each(|(idx, (meta, trace, univariate_skip, z))| {
            let evals = if univariate_skip.skip_rounds > 0 {
                &piop.air.univariate_eval_sumcheck.evals[idx]
            } else {
                &piop.air.regular_sumcheck.evals[idx]
            };
            let (local, next) = evals.split_at(meta.width);
            let mut eq_z = crate::eq_poly(z, Challenge::ONE);
            assert_eq!(trace.columnwise_dot_product(&eq_z), local);
            eq_z.rotate_right(1);
            assert_eq!(trace.columnwise_dot_product(&eq_z), next);
        });

    Proof { log_heights, piop }
}

fn prove_piop<Val, Challenge, A>(
    pk: &ProvingKey,
    inputs: &[VerifierInput<Val, A>],
    traces: Vec<impl Matrix<Val>>,
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<Vec<Challenge>>, PiopProof<Challenge>)
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverInteractionFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverInteractionFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    let traces = traces
        .into_par_iter()
        .map(AirTrace::new)
        .collect::<Vec<_>>();

    let beta: Challenge = challenger.sample_algebra_element();
    let gamma: Challenge = challenger.sample_algebra_element();
    let beta_powers = beta
        .powers()
        .skip(1)
        .take(pk.max_field_count())
        .collect_vec();
    let gamma_powers = gamma
        .powers()
        .skip(1)
        .take(pk.max_bus_index() + 1)
        .collect_vec();

    let (z_fs, fractional_sum) = prove_fractional_sum::<Val, Challenge, _>(
        pk,
        inputs,
        &traces,
        &beta_powers,
        &gamma_powers,
        &mut challenger,
    );
    let claims_fs = fractional_sum
        .sumchecks
        .last()
        .map(|sumcheck| sumcheck.evals.as_slice())
        .unwrap_or(&[]);

    let (zs, air) = prove_air(
        pk,
        inputs,
        traces,
        &beta_powers,
        &gamma_powers,
        &z_fs,
        claims_fs,
        &mut challenger,
    );

    (
        zs,
        PiopProof {
            fractional_sum,
            air,
        },
    )
}

fn prove_fractional_sum<Val, Challenge, A>(
    pk: &ProvingKey,
    inputs: &[VerifierInput<Val, A>],
    traces: &[AirTrace<Val, Challenge>],
    beta_powers: &[Challenge],
    gamma_powers: &[Challenge],
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<Challenge>, FractionalSumProof<Challenge>)
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverInteractionFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverInteractionFolderOnExtension<'t, Val, Challenge>>,
{
    // TODO: Prove fractional-sum check and get real z and evals.

    let trace_fs = izip!(pk.metas(), inputs, traces)
        .map(|(meta, input, trace)| {
            if meta.interaction_count == 0 {
                return AirTrace::default();
            }

            match trace {
                AirTrace::Packing(trace) => {
                    let width = (1 + Challenge::DIMENSION) * meta.interaction_count;
                    let mut trace_fs = RowMajorMatrix::new(
                        vec![Val::Packing::ZERO; width * trace.height()],
                        width,
                    );
                    trace_fs
                        .par_rows_mut()
                        .zip(trace.par_row_slices())
                        .for_each(|(row_fs, row)| {
                            let (numers, denoms) = row_fs.split_at_mut(meta.interaction_count);
                            let denoms = unsafe {
                                transmute::<&mut [_], &mut [Challenge::ExtensionPacking]>(denoms)
                            };
                            let mut builder = ProverInteractionFolderOnPacking {
                                main: RowMajorMatrixView::new(row, meta.width),
                                public_values: input.public_values(),
                                beta_powers,
                                gamma_powers,
                                numers,
                                denoms,
                                interaction_index: 0,
                            };
                            input.air().eval(&mut builder);
                        });
                    AirTrace::Packing(trace_fs)
                }
                AirTrace::Extension(trace) => {
                    let mut trace_fs = RowMajorMatrix::new(
                        vec![Challenge::ZERO; 2 * meta.interaction_count * trace.height()],
                        2 * meta.interaction_count,
                    );
                    trace_fs
                        .par_rows_mut()
                        .zip(trace.par_row_slices())
                        .for_each(|(row_fs, row)| {
                            let (numers, denoms) = row_fs.split_at_mut(meta.interaction_count);
                            let mut builder = ProverInteractionFolderOnExtension {
                                main: RowMajorMatrixView::new(row, meta.width),
                                public_values: input.public_values(),
                                beta_powers,
                                gamma_powers,
                                numers,
                                denoms,
                                interaction_index: 0,
                            };
                            input.air().eval(&mut builder);
                        });
                    AirTrace::Extension(trace_fs)
                }
                _ => unimplemented!(),
            }
        })
        .collect_vec();

    let z = (0..itertools::max(traces.iter().map(|trace| trace.log_height())).unwrap())
        .map(|_| challenger.sample_algebra_element())
        .collect_vec();

    let evals = izip!(pk.metas(), trace_fs)
        .map(|(meta, trace_fs)| {
            if meta.interaction_count == 0 {
                return Vec::new();
            }

            let eq_z = eq_poly(z.rslice(trace_fs.log_height()), Challenge::ONE);
            let evals = trace_fs.columnwise_dot_product(&eq_z);
            if let AirTrace::Packing(_) = trace_fs {
                let (numers, denoms) = evals.split_at(meta.interaction_count);
                chain![
                    cloned(numers),
                    denoms.chunks(Challenge::DIMENSION).map(|chunk| dot_product(
                        cloned(chunk),
                        (0..Challenge::DIMENSION).map(|i| Challenge::ith_basis_element(i).unwrap())
                    ))
                ]
                .collect()
            } else {
                evals
            }
        })
        .collect_vec();

    (
        z,
        FractionalSumProof {
            sums: Vec::new(),
            sumchecks: vec![BatchSumcheckProof {
                compressed_round_polys: Vec::new(),
                evals,
            }],
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_air<Val, Challenge, A>(
    pk: &ProvingKey,
    inputs: &[VerifierInput<Val, A>],
    mut traces: Vec<AirTrace<Val, Challenge>>,
    beta_powers: &[Challenge],
    gamma_powers: &[Challenge],
    z_fs: &[Challenge],
    claims_fs: &[Vec<Challenge>],
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<Vec<Challenge>>, AirProof<Challenge>)
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    let skip_rounds = traces
        .iter()
        .map(|trace| {
            // TODO: Find a better way to choose the optimal rounds to skip automatically.
            const SKIP_ROUNDS: usize = 6;
            (trace.log_height() >= SKIP_ROUNDS + log2_strict_usize(Val::Packing::WIDTH))
                .then_some(SKIP_ROUNDS)
                .unwrap_or_default()
        })
        .collect_vec();
    let regular_rounds = izip!(&traces, cloned(&skip_rounds))
        .map(|(trace, skip_rounds)| trace.log_height() - skip_rounds)
        .collect_vec();
    let max_skip_rounds = itertools::max(cloned(&skip_rounds)).unwrap();
    let max_regular_rounds = itertools::max(cloned(&regular_rounds)).unwrap();

    let r = (0..max_regular_rounds)
        .map(|_| challenger.sample_algebra_element())
        .collect_vec();

    let alpha = challenger.sample_algebra_element::<Challenge>();
    let alpha_powers = {
        let max_alpha_power_count = pk.max_alpha_power_count();
        let mut alpha_powers = alpha.powers().take(max_alpha_power_count).collect_vec();
        alpha_powers.reverse();
        alpha_powers
    };

    let mut univariate_skip_provers = izip!(pk.metas(), inputs, &mut traces, cloned(&skip_rounds))
        .map(|(meta, input, trace, skip_rounds)| {
            (skip_rounds > 0).then(|| {
                UnivariateSkipProver::new(
                    meta,
                    input.air(),
                    input.public_values(),
                    beta_powers,
                    gamma_powers,
                    alpha_powers.rslice(meta.alpha_power_count()),
                    mem::take(trace),
                    skip_rounds,
                )
            })
        })
        .collect_vec();

    let univariate_skips = izip!(
        univariate_skip_provers.iter_mut(),
        cloned(&skip_rounds),
        cloned(&regular_rounds),
    )
    .map(|(prover, skip_rounds, regular_rounds)| {
        prover
            .as_mut()
            .map(|prover| {
                let (zero_check_round_poly, eval_check_round_poly) = prover
                    .compute_round_poly(r.rslice(regular_rounds), z_fs.rslice(regular_rounds));
                UnivariateSkipProof {
                    skip_rounds,
                    zero_check_round_poly,
                    eval_check_round_poly,
                }
            })
            .unwrap_or_default()
    })
    .collect_vec();

    univariate_skips.iter().for_each(|univariate_skip| {
        challenger.observe(Val::from_u8(univariate_skip.skip_rounds as u8));
        cloned(&univariate_skip.zero_check_round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
        cloned(&univariate_skip.eval_check_round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
    });

    let x = challenger.sample_algebra_element();

    let eq_r_helper = EqHelper::new(&r);
    let eq_z_fs_helper = EqHelper::new(z_fs.rslice(max_regular_rounds));

    let mut regular_provers = izip!(
        pk.metas(),
        inputs,
        &mut traces,
        &univariate_skip_provers,
        claims_fs
    )
    .map(|(meta, input, trace, univariate_skip_prover, claims_fs)| {
        if let Some(univariate_skip_prover) = univariate_skip_prover {
            univariate_skip_prover.to_regular_prover(
                x,
                &eq_r_helper,
                &eq_z_fs_helper,
                max_regular_rounds,
            )
        } else {
            let eval_check_claim = dot_product::<Challenge, _, _>(
                cloned(claims_fs),
                cloned(alpha_powers.rslice(2 * meta.interaction_count)),
            );
            RegularSumcheckProver::new(
                meta,
                input.air(),
                input.public_values(),
                Challenge::ZERO,
                eval_check_claim,
                mem::take(trace),
                beta_powers,
                gamma_powers,
                alpha_powers.rslice(meta.alpha_power_count()),
                &eq_r_helper,
                &eq_z_fs_helper,
                max_regular_rounds,
            )
        }
    })
    .collect_vec();

    let delta = challenger.sample_algebra_element::<Challenge>();

    let (z, regular_sumcheck) = {
        let (z, compressed_round_polys) = (0..max_regular_rounds)
            .map(|round| {
                let compressed_round_polys = regular_provers
                    .iter_mut()
                    .map(|state| state.compute_round_poly(round))
                    .collect_vec();

                let compressed_round_poly =
                    CompressedRoundPoly::random_linear_combine(compressed_round_polys, delta);

                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();

                regular_provers
                    .iter_mut()
                    .for_each(|state| state.fix_var(round, z_i));

                (z_i, compressed_round_poly)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        let evals = regular_provers
            .into_iter()
            .map(|state| state.into_evals())
            .collect_vec();

        cloned(evals.iter().flatten()).for_each(|eval| challenger.observe_algebra_element(eval));

        (
            z,
            BatchSumcheckProof {
                compressed_round_polys,
                evals,
            },
        )
    };

    let eta: Challenge = challenger.sample_algebra_element();
    let theta: Challenge = challenger.sample_algebra_element();
    let eta_powers = {
        let max_width = pk.max_width();
        let mut eta_powers = eta.powers().take(2 * max_width).collect_vec();
        eta_powers.reverse();
        eta_powers
    };

    let mut eval_provers = izip!(
        pk.metas(),
        univariate_skip_provers,
        &regular_sumcheck.evals,
        cloned(&regular_rounds)
    )
    .map(|(meta, prover, evals, regular_rounds)| {
        prover.map(|prover| {
            prover.into_univariate_eval_prover(
                x,
                z.rslice(regular_rounds),
                evals,
                eta_powers.rslice(2 * meta.width),
                max_skip_rounds,
            )
        })
    })
    .collect_vec();

    let (z_prime, univariate_eval_sumcheck) = {
        let (z_pirme, compressed_round_polys) = (0..max_skip_rounds)
            .map(|round| {
                let compressed_round_polys = eval_provers
                    .iter_mut()
                    .map(|prover| {
                        prover
                            .as_mut()
                            .map(|prover| prover.compute_round_poly(round))
                            .unwrap_or_default()
                    })
                    .collect_vec();

                let compressed_round_poly =
                    CompressedRoundPoly::random_linear_combine(compressed_round_polys, theta);

                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_prime_i = challenger.sample_algebra_element();

                eval_provers
                    .iter_mut()
                    .flatten()
                    .for_each(|prover| prover.fix_var(round, z_prime_i));

                (z_prime_i, compressed_round_poly)
            })
            .collect::<(Vec<_>, Vec<_>)>();
        let evals = eval_provers
            .into_iter()
            .map(|prover| prover.map(|prover| prover.into_evals()).unwrap_or_default())
            .collect_vec();

        cloned(evals.iter().flatten()).for_each(|eval| challenger.observe_algebra_element(eval));

        (
            z_pirme,
            BatchSumcheckProof {
                compressed_round_polys,
                evals,
            },
        )
    };

    let zs = izip!(skip_rounds, regular_rounds)
        .map(|(skip_rounds, regular_rounds)| {
            chain![z_prime.rslice(skip_rounds), z.rslice(regular_rounds)]
                .copied()
                .collect()
        })
        .collect();

    (
        zs,
        AirProof {
            univariate_skips,
            regular_sumcheck,
            univariate_eval_sumcheck,
        },
    )
}
