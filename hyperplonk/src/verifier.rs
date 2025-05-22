use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;

use itertools::{Itertools, chain, cloned, enumerate, izip};
use p3_air::Air;
use p3_air_ext::VerifierInput;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_ml_pcs::{MlPcs, MlQuery, eq_eval};
use tracing::instrument;

use crate::{
    AirMeta, AirProof, EvalClaim, FS_ARITY, Fraction, FractionalSumProof, HyperPlonkGenericConfig,
    PcsError, PiopProof, Proof, RSlice, Val, VerifierConstraintFolder, VerifyingKey,
    eval_fractional_sum_accumulator, evaluate_ml_poly, evaluations_on_domain, fix_var,
    lagrange_evals, random_linear_combine, selectors_at_point,
};

#[derive(Debug)]
pub enum VerificationError<PcsError> {
    InvalidProofShape,
    NonZeroFractionalSum,
    UnivariateSkipEvaluationMismatch,
    OodEvaluationMismatch,
    InvalidOpeningArgument(PcsError),
}

#[instrument(skip_all)]
pub fn verify<C, A>(
    config: &C,
    vk: &VerifyingKey,
    inputs: Vec<VerifierInput<Val<C>, A>>,
    proof: &Proof<C>,
) -> Result<(), VerificationError<PcsError<C>>>
where
    C: HyperPlonkGenericConfig,
    Val<C>: TwoAdicField + Ord,
    A: for<'t> Air<VerifierConstraintFolder<'t, Val<C>, C::Challenge>>,
{
    debug_assert!(!inputs.is_empty());

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    cloned(&proof.log_bs).for_each(|log_b| challenger.observe(Val::<C>::from_u8(log_b as u8)));
    challenger.observe(proof.commitment.clone());
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    let zs = verify_piop(vk, &inputs, &proof.log_bs, &proof.piop, &mut challenger)?;

    pcs.verify(
        vec![(
            proof.commitment.clone(),
            queries_and_evals(vk.metas(), &proof.piop.air, &zs),
        )],
        &proof.pcs,
        &mut challenger,
    )
    .map_err(VerificationError::InvalidOpeningArgument)?;

    Ok(())
}

fn verify_piop<Val, Challenge, A, PcsError>(
    vk: &VerifyingKey,
    inputs: &[VerifierInput<Val, A>],
    log_bs: &[usize],
    proof: &PiopProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Vec<Challenge>>, VerificationError<PcsError>>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<VerifierConstraintFolder<'t, Val, Challenge>>,
{
    let beta: Challenge = challenger.sample_algebra_element();
    let gamma: Challenge = challenger.sample_algebra_element();
    let beta_powers = beta
        .powers()
        .skip(1)
        .take(vk.max_field_count())
        .collect_vec();
    let gamma_powers = gamma
        .powers()
        .skip(1)
        .take(vk.max_bus_index() + 1)
        .collect_vec();

    let claims_fs = verify_fractional_sum(vk, log_bs, &proof.fractional_sum, &mut challenger)?;

    let zs = verify_air(
        vk,
        inputs,
        log_bs,
        &beta_powers,
        &gamma_powers,
        &claims_fs,
        &proof.air,
        &mut challenger,
    )?;

    Ok(zs)
}

fn verify_fractional_sum<Val, Challenge, PcsError>(
    vk: &VerifyingKey,
    log_bs: &[usize],
    proof: &FractionalSumProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<EvalClaim<Challenge>>, VerificationError<PcsError>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
{
    if !vk.has_any_interaction() {
        return Ok(vec![Default::default(); vk.metas.len()]);
    }

    let max_log_b = itertools::max(cloned(log_bs)).unwrap();

    if proof.sums.len() != vk.metas().len()
        || !izip!(vk.metas(), &proof.sums).all(|(meta, sums)| sums.len() == meta.interaction_count)
        || proof.layers.len() != max_log_b
        || !proof.layers.iter().enumerate().all(|(rounds, layer)| {
            layer.compressed_round_polys.len() == rounds
                && layer
                    .compressed_round_polys
                    .iter()
                    .all(|compressed_round_poly| compressed_round_poly.0.len() == 1 + FS_ARITY)
                && layer.evals.len() == vk.metas().len()
                && izip!(vk.metas(), cloned(log_bs), &layer.evals).all(|(meta, log_b, evals)| {
                    rounds >= log_b || evals.len() == 2 * meta.interaction_count * FS_ARITY
                })
        })
    {
        return Err(VerificationError::InvalidProofShape);
    }

    if Fraction::sum(proof.sums.iter().flatten()) != Some(Challenge::ZERO) {
        return Err(VerificationError::NonZeroFractionalSum);
    }

    proof.sums.iter().flatten().for_each(|fraction| {
        challenger.observe_algebra_element(fraction.numer);
        challenger.observe_algebra_element(fraction.denom);
    });

    let mut claims = proof
        .sums
        .iter()
        .map(|sums| EvalClaim {
            z: Vec::new(),
            evals: sums.iter().flat_map(|sum| [sum.numer, sum.denom]).collect(),
        })
        .collect_vec();
    proof
        .layers
        .iter()
        .enumerate()
        .try_for_each(|(rounds, layer)| {
            let alpha: Challenge = challenger.sample_algebra_element();
            let beta: Challenge = challenger.sample_algebra_element();

            let mut z = {
                let mut claim = izip!(cloned(log_bs), &claims, beta.powers())
                    .filter(|(log_b, _, _)| *log_b > rounds)
                    .map(|(_, claim, beta_power)| {
                        random_linear_combine(&claim.evals, alpha) * beta_power
                    })
                    .sum::<Challenge>();
                let z = layer
                    .compressed_round_polys
                    .iter()
                    .map(|compressed_round_poly| {
                        cloned(&compressed_round_poly.0)
                            .for_each(|coeff| challenger.observe_algebra_element(coeff));
                        let z_i = challenger.sample_algebra_element();
                        claim = compressed_round_poly.subclaim(claim, z_i);
                        z_i
                    })
                    .collect_vec();

                cloned(layer.evals.iter().flatten())
                    .for_each(|eval| challenger.observe_algebra_element(eval));

                let eval = izip!(&claims, &layer.evals, beta.powers())
                    .map(|(claim, evals, beta_power)| {
                        if evals.is_empty() {
                            return Challenge::ZERO;
                        }
                        eval_fractional_sum_accumulator(FS_ARITY, evals, alpha)
                            * eq_eval(&claim.z, &z)
                            * beta_power
                    })
                    .sum::<Challenge>();
                if eval != claim {
                    return Err(VerificationError::OodEvaluationMismatch);
                }

                z
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

            Ok(())
        })?;

    Ok(claims)
}

#[allow(clippy::too_many_arguments)]
fn verify_air<Val, Challenge, A, PcsError>(
    vk: &VerifyingKey,
    inputs: &[VerifierInput<Val, A>],
    log_bs: &[usize],
    beta_powers: &[Challenge],
    gamma_powers: &[Challenge],
    claims_fs: &[EvalClaim<Challenge>],
    proof: &AirProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Vec<Challenge>>, VerificationError<PcsError>>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<VerifierConstraintFolder<'t, Val, Challenge>>,
{
    let skip_rounds = proof
        .univariate_skips
        .iter()
        .map(|univariate_skip| univariate_skip.skip_rounds)
        .collect_vec();
    let regular_rounds = izip!(cloned(log_bs), cloned(&skip_rounds))
        .map(|(log_b, skip_rounds)| log_b.saturating_sub(skip_rounds))
        .collect_vec();
    let max_skip_rounds = itertools::max(cloned(&skip_rounds)).unwrap();
    let max_regular_rounds = itertools::max(cloned(&regular_rounds)).unwrap();

    if proof.univariate_skips.len() != vk.metas().len()
        || !izip!(vk.metas(), cloned(log_bs), cloned(&proof.univariate_skips)).all(
            |(meta, log_b, univariate_skip)| {
                univariate_skip.skip_rounds <= log_b
                    && (if univariate_skip.skip_rounds == 0 {
                        univariate_skip.zero_check_round_poly.0.is_empty()
                            && univariate_skip.eval_check_round_poly.0.is_empty()
                    } else {
                        univariate_skip.zero_check_round_poly.0.len()
                            == meta.zero_check_uv_degree.saturating_sub(1)
                                << univariate_skip.skip_rounds
                            && univariate_skip.eval_check_round_poly.0.len()
                                == meta.eval_check_uv_degree << univariate_skip.skip_rounds
                    })
            },
        )
        || proof.regular.compressed_round_polys.len() != max_regular_rounds
        || !proof
            .regular
            .compressed_round_polys
            .iter()
            .all(|compressed_round_poly| {
                compressed_round_poly.0.len() == vk.max_regular_sumcheck_degree() + 1
            })
        || proof.regular.evals.len() != vk.metas().len()
        || !izip!(vk.metas(), &proof.regular.evals)
            .all(|(meta, evals)| evals.len() == 2 * meta.width)
        || proof.univariate_eval_check.compressed_round_polys.len() != max_skip_rounds
        || !proof
            .univariate_eval_check
            .compressed_round_polys
            .iter()
            .all(|compressed_round_poly| compressed_round_poly.0.len() == 2)
        || proof.univariate_eval_check.evals.len() != vk.metas().len()
        || !izip!(
            vk.metas(),
            cloned(&skip_rounds),
            &proof.univariate_eval_check.evals
        )
        .all(|(meta, skip_rounds, evals)| skip_rounds == 0 || evals.len() == 2 * meta.width)
    {
        return Err(VerificationError::InvalidProofShape);
    }

    let z_zc = (0..max_regular_rounds)
        .map(|_| challenger.sample_algebra_element::<Challenge>())
        .collect_vec();

    let alpha: Challenge = challenger.sample_algebra_element();

    proof.univariate_skips.iter().for_each(|univariate_skip| {
        challenger.observe(Val::from_u8(univariate_skip.skip_rounds as u8));
        cloned(&univariate_skip.zero_check_round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
        cloned(&univariate_skip.eval_check_round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
    });

    let x: Challenge = challenger.sample_algebra_element();

    let delta: Challenge = challenger.sample_algebra_element();

    let z_regular = {
        let claims = izip!(
            vk.metas(),
            claims_fs,
            &proof.univariate_skips,
            delta.powers()
        )
        .map(|(meta, claim_fs, univariate_skip, delta_power)| {
            let eval_check_claim = random_linear_combine(&claim_fs.evals, alpha);
            let claim = if univariate_skip.skip_rounds == 0 {
                eval_check_claim
            } else {
                if meta.interaction_count > 0 {
                    let evaluations = evaluations_on_domain(
                        univariate_skip.skip_rounds,
                        &univariate_skip.eval_check_round_poly,
                    );
                    let z_fs_skipped = &claim_fs.z[..univariate_skip.skip_rounds];
                    if evaluate_ml_poly(&evaluations, z_fs_skipped) != eval_check_claim {
                        return Err(VerificationError::UnivariateSkipEvaluationMismatch);
                    }
                }
                univariate_skip.zero_check_round_poly.subclaim(x)
                    * (x.exp_power_of_2(univariate_skip.skip_rounds) - Val::ONE)
                    + univariate_skip.eval_check_round_poly.subclaim(x)
            };
            Ok(claim * delta_power)
        })
        .try_collect::<_, Vec<_>, _>()?;

        let claims_at_round = |round| {
            izip!(&regular_rounds, &claims)
                .filter(|(regular_rounds, _)| max_regular_rounds == *regular_rounds + round)
                .map(|(_, subclaim)| *subclaim)
                .sum::<Challenge>()
        };

        let mut claim = claims_at_round(0);
        let z = enumerate(&proof.regular.compressed_round_polys)
            .map(|(round, compressed_round_poly)| {
                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();
                claim = compressed_round_poly.subclaim(claim, z_i) + claims_at_round(round + 1);
                z_i
            })
            .collect_vec();

        cloned(proof.regular.evals.iter().flatten())
            .for_each(|eval| challenger.observe_algebra_element(eval));

        let eval = izip!(
            vk.metas(),
            inputs,
            cloned(&skip_rounds),
            cloned(&regular_rounds),
            claims_fs,
            &proof.regular.evals,
            delta.powers(),
        )
        .map(
            |(meta, input, skip_rounds, regular_rounds, claim_fs, evals, delta_power)| {
                let sels = selectors_at_point(skip_rounds, x);
                let z_fs = meta
                    .has_interaction()
                    .then(|| claim_fs.z.rslice(regular_rounds));
                let z_zc = z_zc.rslice(regular_rounds);
                let z = z.rslice(regular_rounds);
                let eq_0_z = eq_eval(repeat_n(&Challenge::ZERO, z.len()), z);
                let eq_1_z = eq_eval(repeat_n(&Challenge::ONE, z.len()), z);
                let is_first_row = sels.is_first_row * eq_0_z;
                let is_last_row = sels.is_last_row * eq_1_z;
                let is_transition = Challenge::ONE - eq_1_z + sels.is_transition * eq_1_z;
                let mut builder = VerifierConstraintFolder {
                    main: DenseMatrix::new(evals, meta.width),
                    public_values: input.public_values(),
                    is_first_row,
                    is_last_row,
                    is_transition,
                    alpha,
                    beta_powers,
                    gamma_powers,
                    zero_check_accumulator: Challenge::ZERO,
                    eval_check_accumulator: Challenge::ZERO,
                };
                input.air().eval(&mut builder);
                let zero_check_eval = builder.zero_check_accumulator
                    * alpha.exp_u64(2 * meta.interaction_count as u64)
                    * eq_eval(z_zc, z);
                let eval_check_eval = z_fs
                    .map(|z_fs| builder.eval_check_accumulator * eq_eval(z_fs, z))
                    .unwrap_or_default();
                (zero_check_eval + eval_check_eval) * delta_power
            },
        )
        .sum::<Challenge>();
        if eval != claim {
            return Err(VerificationError::OodEvaluationMismatch);
        }

        z
    };

    let eta: Challenge = challenger.sample_algebra_element();
    let theta: Challenge = challenger.sample_algebra_element();

    let z_skip = {
        let claims = izip!(cloned(&skip_rounds), &proof.regular.evals, theta.powers())
            .map(|(skip_rounds, evals, theta_power)| {
                if skip_rounds != 0 {
                    random_linear_combine(evals, eta) * theta_power
                } else {
                    Challenge::ZERO
                }
            })
            .collect_vec();

        let claim_at_round = |round| {
            izip!(&skip_rounds, &claims)
                .filter(|(skip_rounds, _)| max_skip_rounds == *skip_rounds + round)
                .map(|(_, subclaim)| *subclaim)
                .sum::<Challenge>()
        };

        let mut claim = claim_at_round(0);
        let z = enumerate(&proof.univariate_eval_check.compressed_round_polys)
            .map(|(round, compressed_round_poly)| {
                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();
                claim = compressed_round_poly.subclaim(claim, z_i) + claim_at_round(round + 1);
                z_i
            })
            .collect_vec();

        cloned(proof.univariate_eval_check.evals.iter().flatten())
            .for_each(|eval| challenger.observe_algebra_element(eval));

        let eval = izip!(
            cloned(&skip_rounds),
            &proof.univariate_eval_check.evals,
            theta.powers()
        )
        .map(|(skip_rounds, evals, theta_power)| {
            random_linear_combine(evals, eta)
                * evaluate_ml_poly(&lagrange_evals(skip_rounds, x), z.rslice(skip_rounds))
                * theta_power
        })
        .sum::<Challenge>();
        if eval != claim {
            return Err(VerificationError::OodEvaluationMismatch);
        }

        z
    };

    let zs = izip!(skip_rounds, regular_rounds)
        .map(|(skip_rounds, regular_rounds)| {
            chain![z_skip.rslice(skip_rounds), z_regular.rslice(regular_rounds)]
                .copied()
                .collect()
        })
        .collect();

    Ok(zs)
}

pub(crate) fn queries_and_evals<Challenge: Clone>(
    metas: &[AirMeta],
    proof: &AirProof<Challenge>,
    zs: &[Vec<Challenge>],
) -> Vec<Vec<(MlQuery<Challenge>, Vec<Challenge>)>> {
    izip!(metas, &proof.univariate_skips, zs)
        .enumerate()
        .map(|(idx, (meta, univariate_skip, z))| {
            let evals = if univariate_skip.skip_rounds > 0 {
                &proof.univariate_eval_check.evals[idx]
            } else {
                &proof.regular.evals[idx]
            };
            let (local, next) = evals.split_at(meta.width);
            vec![
                (MlQuery::Eq(z.to_vec()), local.to_vec()),
                (MlQuery::EqRotateRight(z.to_vec(), 1), next.to_vec()),
            ]
        })
        .collect()
}
