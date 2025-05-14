use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use itertools::{Itertools, chain, cloned, izip, rev};
use p3_air::Air;
use p3_air_ext::{ProverInput, VerifierInput};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField, dot_product};
use p3_matrix::Matrix;
use p3_matrix::dense::DenseMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{
    AirProof, AirRegularProver, AirUnivariateSkipProof, AirUnivariateSkipProver,
    BatchSumcheckProof, CompressedRoundPoly, EqHelper, EvalClaim, FractionalSumProof,
    FractionalSumRegularProver, PiopProof, Proof, ProverConstraintFolderOnExtension,
    ProverConstraintFolderOnExtensionPacking, ProverConstraintFolderOnPacking,
    ProverInteractionFolderOnExtension, ProverInteractionFolderOnPacking, ProvingKey, RSlice,
    Trace, fix_var, fractional_sum_layers, fractional_sum_trace,
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
    debug_assert!(!inputs.is_empty());

    #[cfg(feature = "check-constraints")]
    crate::check_constraints(&inputs);

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
                &piop.air.univariate_eval_check.evals[idx]
            } else {
                &piop.air.regular.evals[idx]
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
    let traces = info_span!("pack traces")
        .in_scope(|| traces.into_iter().map(Trace::new).collect::<Vec<_>>());

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

    let (claims_fs, fractional_sum) = prove_fractional_sum::<Val, Challenge, _>(
        pk,
        inputs,
        &traces,
        &beta_powers,
        &gamma_powers,
        &mut challenger,
    );

    let (zs, air) = prove_air(
        pk,
        inputs,
        traces,
        &beta_powers,
        &gamma_powers,
        &claims_fs,
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

#[instrument(skip_all)]
fn prove_fractional_sum<Val, Challenge, A>(
    pk: &ProvingKey,
    inputs: &[VerifierInput<Val, A>],
    traces: &[Trace<Val, Challenge>],
    beta_powers: &[Challenge],
    gamma_powers: &[Challenge],
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<EvalClaim<Challenge>>, FractionalSumProof<Challenge>)
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverInteractionFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverInteractionFolderOnExtension<'t, Val, Challenge>>,
{
    if !pk.has_any_interaction() {
        return (
            vec![Default::default(); pk.metas().len()],
            Default::default(),
        );
    }

    let (mut layers, sums) = izip!(pk.metas(), inputs, traces)
        .map(|(meta, input, trace)| {
            fractional_sum_trace(meta, input, trace, beta_powers, gamma_powers)
                .map(|input_layer| fractional_sum_layers(meta.interaction_count, input_layer))
                .unwrap_or_default()
        })
        .collect::<(Vec<_>, Vec<_>)>();

    sums.iter().flatten().for_each(|frac| {
        challenger.observe_algebra_element(frac.numer);
        challenger.observe_algebra_element(frac.denom);
    });

    let max_interaction_count = pk.max_interaction_count();
    let max_log_height = itertools::max(traces.iter().map(|trace| trace.log_height())).unwrap();

    let mut claims = sums
        .iter()
        .map(|sums| EvalClaim {
            z: Vec::new(),
            evals: sums
                .iter()
                .flat_map(|sum| [sum.numer, sum.denom])
                .collect_vec(),
        })
        .collect_vec();
    let layers = (0..max_log_height)
        .map(|rounds| {
            let alpha: Challenge = challenger.sample_algebra_element();
            let beta: Challenge = challenger.sample_algebra_element();
            let alpha_powers = {
                let mut alpha_powers = alpha.powers().take(2 * max_interaction_count).collect_vec();
                alpha_powers.reverse();
                alpha_powers
            };

            let mut provers = izip!(pk.metas(), &mut layers, &claims)
                .map(|(meta, layers, claim)| {
                    layers.pop().map(|layer| {
                        let alpha_powers = alpha_powers.rslice(2 * meta.interaction_count);
                        FractionalSumRegularProver::new(
                            meta.interaction_count,
                            dot_product(cloned(&claim.evals), cloned(alpha_powers)),
                            layer,
                            alpha_powers,
                            &claim.z,
                        )
                    })
                })
                .collect_vec();

            let (mut z, layer) = {
                let (z, compressed_round_polys) = rev(0..rounds)
                    .map(|log_b| {
                        let compressed_round_polys = provers
                            .iter_mut()
                            .map(|prover| {
                                prover
                                    .as_mut()
                                    .map(|prover| prover.compute_round_poly(log_b))
                                    .unwrap_or_default()
                            })
                            .collect_vec();

                        let compressed_round_poly = CompressedRoundPoly::random_linear_combine(
                            compressed_round_polys,
                            beta,
                        );

                        cloned(&compressed_round_poly.0)
                            .for_each(|coeff| challenger.observe_algebra_element(coeff));
                        let z_i = challenger.sample_algebra_element();

                        provers.iter_mut().for_each(|prover| {
                            if let Some(prover) = prover.as_mut() {
                                prover.fix_var(log_b, z_i)
                            }
                        });

                        (z_i, compressed_round_poly)
                    })
                    .collect::<(Vec<_>, Vec<_>)>();

                let evals = provers
                    .into_iter()
                    .map(|prover| prover.map(|prover| prover.into_evals()).unwrap_or_default())
                    .collect_vec();

                cloned(evals.iter().flatten())
                    .for_each(|eval| challenger.observe_algebra_element(eval));

                (
                    z,
                    BatchSumcheckProof {
                        compressed_round_polys,
                        evals,
                    },
                )
            };

            let z_first = challenger.sample_algebra_element();
            z.insert(0, z_first);

            izip!(&mut claims, &layer.evals).for_each(|(claim, evals)| {
                if evals.is_empty() {
                    return;
                }
                claim.evals = fix_var(DenseMatrix::new(evals, evals.len() / 2), z_first).values;
                claim.z = z.clone();
            });

            layer
        })
        .collect();

    (claims, FractionalSumProof { sums, layers })
}

#[instrument(skip_all)]
fn prove_air<Val, Challenge, A>(
    pk: &ProvingKey,
    inputs: &[VerifierInput<Val, A>],
    mut traces: Vec<Trace<Val, Challenge>>,
    beta_powers: &[Challenge],
    gamma_powers: &[Challenge],
    claims_fs: &[EvalClaim<Challenge>],
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

    let z_zc = (0..max_regular_rounds)
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
                AirUnivariateSkipProver::new(
                    meta,
                    input.air(),
                    input.public_values(),
                    mem::take(trace),
                    beta_powers,
                    gamma_powers,
                    alpha_powers.rslice(meta.alpha_power_count()),
                    skip_rounds,
                )
            })
        })
        .collect_vec();

    let univariate_skips = izip!(
        pk.metas(),
        univariate_skip_provers.iter_mut(),
        cloned(&skip_rounds),
        cloned(&regular_rounds),
        claims_fs,
    )
    .map(|(meta, prover, skip_rounds, regular_rounds, claim_fs)| {
        prover
            .as_mut()
            .map(|prover| {
                let (zero_check_round_poly, eval_check_round_poly) = prover.compute_round_poly(
                    z_zc.rslice(regular_rounds),
                    meta.has_interaction()
                        .then(|| claim_fs.z.rslice(regular_rounds))
                        .unwrap_or_default(),
                );
                AirUnivariateSkipProof {
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

    let zero_check_eq_helper = EqHelper::new(&z_zc);

    let mut regular_provers = izip!(
        pk.metas(),
        inputs,
        &mut traces,
        &univariate_skip_provers,
        claims_fs,
    )
    .map(|(meta, input, trace, univariate_skip_prover, claim_fs)| {
        if let Some(univariate_skip_prover) = univariate_skip_prover {
            univariate_skip_prover.to_regular_prover(x, &zero_check_eq_helper, &claim_fs.z)
        } else {
            let eval_check_claim = dot_product::<Challenge, _, _>(
                cloned(&claim_fs.evals),
                cloned(alpha_powers.rslice(2 * meta.interaction_count)),
            );
            AirRegularProver::new(
                meta,
                input.air(),
                input.public_values(),
                mem::take(trace),
                beta_powers,
                gamma_powers,
                alpha_powers.rslice(meta.alpha_power_count()),
                Challenge::ZERO,
                eval_check_claim,
                &zero_check_eq_helper,
                &claim_fs.z,
            )
        }
    })
    .collect_vec();

    let delta = challenger.sample_algebra_element::<Challenge>();

    let (z_regular, regular) = {
        let (z, compressed_round_polys) = rev(0..max_regular_rounds)
            .map(|log_b| {
                let compressed_round_polys = regular_provers
                    .iter_mut()
                    .map(|prover| prover.compute_round_poly(log_b))
                    .collect_vec();

                let compressed_round_poly =
                    CompressedRoundPoly::random_linear_combine(compressed_round_polys, delta);

                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();

                regular_provers
                    .iter_mut()
                    .for_each(|prover| prover.fix_var(log_b, z_i));

                (z_i, compressed_round_poly)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        let evals = regular_provers
            .into_iter()
            .map(|prover| prover.into_evals())
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
        &regular.evals,
        cloned(&regular_rounds)
    )
    .map(|(meta, prover, evals, regular_rounds)| {
        prover.map(|prover| {
            prover.into_univariate_eval_prover(
                x,
                z_regular.rslice(regular_rounds),
                evals,
                eta_powers.rslice(2 * meta.width),
            )
        })
    })
    .collect_vec();

    let (z_skip, univariate_eval_check) = {
        let (z, compressed_round_polys) = rev(0..max_skip_rounds)
            .map(|log_b| {
                let compressed_round_polys = eval_provers
                    .iter_mut()
                    .map(|prover| {
                        prover
                            .as_mut()
                            .map(|prover| prover.compute_round_poly(log_b))
                            .unwrap_or_default()
                    })
                    .collect_vec();

                let compressed_round_poly =
                    CompressedRoundPoly::random_linear_combine(compressed_round_polys, theta);

                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();

                eval_provers
                    .iter_mut()
                    .flatten()
                    .for_each(|prover| prover.fix_var(log_b, z_i));

                (z_i, compressed_round_poly)
            })
            .collect::<(Vec<_>, Vec<_>)>();
        let evals = eval_provers
            .into_iter()
            .map(|prover| prover.map(|prover| prover.into_evals()).unwrap_or_default())
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

    let zs = izip!(skip_rounds, regular_rounds)
        .map(|(skip_rounds, regular_rounds)| {
            chain![z_skip.rslice(skip_rounds), z_regular.rslice(regular_rounds)]
                .copied()
                .collect()
        })
        .collect();

    (
        zs,
        AirProof {
            univariate_skips,
            regular,
            univariate_eval_check,
        },
    )
}
