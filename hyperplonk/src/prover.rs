use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use itertools::{Itertools, chain, cloned, izip, rev};
use p3_air::Air;
use p3_air_ext::{ProverInput, VerifierInput};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{
    ExtensionField, Field, PackedValue, PrimeCharacteristicRing, TwoAdicField, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::DenseMatrix;
use p3_ml_pcs::MlPcs;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{
    AirProof, AirRegularProver, AirUnivariateSkipProof, AirUnivariateSkipProver,
    BatchSumcheckProof, CompressedRoundPoly, EqHelper, EvalClaim, FractionalSumProof,
    FractionalSumRegularProver, HyperPlonkGenericConfig, PiopProof, Proof,
    ProverConstraintFolderOnExtension, ProverConstraintFolderOnExtensionPacking,
    ProverConstraintFolderOnPacking, ProverInteractionFolderOnExtension,
    ProverInteractionFolderOnPacking, ProvingKey, RSlice, Trace, Val, fix_var,
    fractional_sum_layers, fractional_sum_trace, queries_and_evals,
};

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)]
pub fn prove<
    C,
    #[cfg(feature = "check-constraints")] A: for<'a> Air<crate::DebugConstraintBuilder<'a, Val<C>>>,
    #[cfg(not(feature = "check-constraints"))] A,
>(
    config: &C,
    pk: &ProvingKey,
    inputs: Vec<ProverInput<Val<C>, A>>,
) -> Proof<C>
where
    C: HyperPlonkGenericConfig,
    Val<C>: TwoAdicField + Ord,
    A: for<'t> Air<ProverInteractionFolderOnPacking<'t, Val<C>, C::Challenge>>
        + for<'t> Air<ProverInteractionFolderOnExtension<'t, Val<C>, C::Challenge>>
        + for<'t> Air<ProverConstraintFolderOnPacking<'t, Val<C>, C::Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtension<'t, Val<C>, C::Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val<C>, C::Challenge>>,
{
    debug_assert!(!inputs.is_empty());

    #[cfg(feature = "check-constraints")]
    crate::check_constraints(&inputs);

    let (inputs, traces) = inputs.into_iter().map_into().collect::<(Vec<_>, Vec<_>)>();
    let log_bs = traces
        .iter()
        .map(|mat| log2_strict_usize(mat.height()))
        .collect_vec();

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    let (commitment, prover_data) =
        info_span!("commit to main data").in_scope(|| pcs.commit(traces));

    cloned(&log_bs).for_each(|log_b| challenger.observe(Val::<C>::from_u8(log_b as u8)));
    challenger.observe(commitment.clone());
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    let (zs, piop) = {
        let traces = (0..pk.metas().len())
            .map(|i| pcs.get_evaluations(&prover_data, i))
            .collect();
        prove_piop(pk, &inputs, traces, &mut challenger)
    };

    let pcs = info_span!("open").in_scope(|| {
        pcs.open(
            vec![(&prover_data, queries_and_evals(pk.metas(), &piop.air, &zs))],
            &mut challenger,
        )
    });

    Proof {
        commitment,
        log_bs,
        piop,
        pcs,
    }
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

    let (sums, mut layers) = info_span!("compute sums and layers").in_scope(|| {
        izip!(pk.metas(), inputs, traces)
            .map(|(meta, input, trace)| {
                fractional_sum_trace(meta, input, trace, beta_powers, gamma_powers)
                    .map(|input_layer| fractional_sum_layers(meta.interaction_count, input_layer))
                    .unwrap_or_default()
            })
            .collect::<(Vec<_>, Vec<_>)>()
    });

    sums.iter().flatten().for_each(|fraction| {
        challenger.observe_algebra_element(fraction.numer);
        challenger.observe_algebra_element(fraction.denom);
    });

    let max_interaction_count = pk.max_interaction_count();
    let max_log_b = itertools::max(traces.iter().map(|trace| trace.log_b())).unwrap();

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
    let layers = (0..max_log_b)
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

            let (mut z, layer) = info_span!("regular rounds").in_scope(|| {
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
            });

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
            if trace.log_b() >= SKIP_ROUNDS + log2_strict_usize(Val::Packing::WIDTH) {
                SKIP_ROUNDS
            } else {
                0
            }
        })
        .collect_vec();
    let regular_rounds = izip!(&traces, cloned(&skip_rounds))
        .map(|(trace, skip_rounds)| trace.log_b() - skip_rounds)
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

    let univariate_skips = info_span!("univariate round").in_scope(|| {
        izip!(
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
                        if meta.has_interaction() {
                            claim_fs.z.rslice(regular_rounds)
                        } else {
                            &[]
                        },
                    );
                    AirUnivariateSkipProof {
                        skip_rounds,
                        zero_check_round_poly,
                        eval_check_round_poly,
                    }
                })
                .unwrap_or_default()
        })
        .collect_vec()
    });

    univariate_skips.iter().for_each(|univariate_skip| {
        challenger.observe(Val::from_u8(univariate_skip.skip_rounds as u8));
        cloned(&univariate_skip.zero_check_round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
        cloned(&univariate_skip.eval_check_round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
    });

    let x = challenger.sample_algebra_element();

    let zero_check_eq_helper = EqHelper::new(&z_zc);

    let mut regular_provers = info_span!("univariate to regular").in_scope(|| {
        izip!(
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
        .collect_vec()
    });

    let delta = challenger.sample_algebra_element::<Challenge>();

    let (z_regular, regular) = info_span!("regular rounds").in_scope(|| {
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
    });

    let eta: Challenge = challenger.sample_algebra_element();
    let theta: Challenge = challenger.sample_algebra_element();
    let eta_powers = {
        let max_width = pk.max_width();
        let mut eta_powers = eta.powers().take(2 * max_width).collect_vec();
        eta_powers.reverse();
        eta_powers
    };

    let mut univariate_eval_provers = info_span!("univariate to univariate eval").in_scope(|| {
        izip!(
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
        .collect_vec()
    });

    let (z_skip, univariate_eval_check) = info_span!("univariate eval rounds").in_scope(|| {
        let (z, compressed_round_polys) = rev(0..max_skip_rounds)
            .map(|log_b| {
                let compressed_round_polys = univariate_eval_provers
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

                univariate_eval_provers
                    .iter_mut()
                    .flatten()
                    .for_each(|prover| prover.fix_var(log_b, z_i));

                (z_i, compressed_round_poly)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        let evals = univariate_eval_provers
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
    });

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
