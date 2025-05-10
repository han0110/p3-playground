use alloc::vec::Vec;
use core::iter::repeat_n;

use itertools::{Itertools, chain, cloned, enumerate, izip};
use p3_air::Air;
use p3_air_ext::VerifierInput;
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrixView;

use crate::{
    AirProof, FractionalSumProof, PiopProof, Proof, RSlice, VerifierConstraintFolder, VerifyingKey,
    eq_eval, evaluate_ml_poly, evaluations_on_domain, lagrange_evals, random_linear_combine,
    selectors_at_point,
};

#[derive(Debug)]
pub enum Error {
    InvalidProofShape,
    UnivariateSkipEvaluationMismatch,
    OodEvaluationMismatch,
}

pub fn verify<Val, Challenge, A>(
    vk: &VerifyingKey,
    inputs: Vec<VerifierInput<Val, A>>,
    proof: &Proof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<(), Error>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<VerifierConstraintFolder<'t, Val, Challenge>>,
{
    assert!(!inputs.is_empty());

    // TODO: Observe commitment.

    cloned(&proof.log_heights)
        .for_each(|log_height| challenger.observe(Val::from_u8(log_height as u8)));
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    let _zs = verify_piop(
        vk,
        &inputs,
        &proof.log_heights,
        &proof.piop,
        &mut challenger,
    )?;

    // TODO: PCS verify.

    Ok(())
}

pub fn verify_piop<Val, Challenge, A>(
    vk: &VerifyingKey,
    inputs: &[VerifierInput<Val, A>],
    log_heights: &[usize],
    proof: &PiopProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Vec<Challenge>>, Error>
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

    let z_fs = verify_fractional_sum(
        vk,
        inputs,
        log_heights,
        &proof.fractional_sum,
        &mut challenger,
    )?;
    let claims_fs = proof
        .fractional_sum
        .sumchecks
        .last()
        .map(|sumcheck| sumcheck.evals.as_slice())
        .unwrap();

    let zs = verify_air(
        vk,
        inputs,
        log_heights,
        &beta_powers,
        &gamma_powers,
        &z_fs,
        claims_fs,
        &proof.air,
        &mut challenger,
    )?;

    Ok(zs)
}

pub fn verify_fractional_sum<Val, Challenge, A>(
    _vk: &VerifyingKey,
    _inputs: &[VerifierInput<Val, A>],
    log_heights: &[usize],
    _proof: &FractionalSumProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Challenge>, Error>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<VerifierConstraintFolder<'t, Val, Challenge>>,
{
    // TODO: Verify fractional sum check and get real z.

    let z = (0..itertools::max(cloned(log_heights)).unwrap())
        .map(|_| challenger.sample_algebra_element())
        .collect_vec();

    Ok(z)
}

#[allow(clippy::too_many_arguments)]
pub fn verify_air<Val, Challenge, A>(
    vk: &VerifyingKey,
    inputs: &[VerifierInput<Val, A>],
    log_heights: &[usize],
    beta_powers: &[Challenge],
    gamma_powers: &[Challenge],
    z_fs: &[Challenge],
    claims_fs: &[Vec<Challenge>],
    proof: &AirProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Vec<Challenge>>, Error>
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
    let regular_rounds = izip!(cloned(log_heights), cloned(&skip_rounds))
        .map(|(log_height, skip_rounds)| log_height.saturating_sub(skip_rounds))
        .collect_vec();
    let max_skip_rounds = itertools::max(cloned(&skip_rounds)).unwrap();
    let max_regular_rounds = itertools::max(cloned(&regular_rounds)).unwrap();

    if proof.univariate_skips.len() != vk.metas().len()
        || !izip!(
            vk.metas(),
            cloned(log_heights),
            cloned(&proof.univariate_skips)
        )
        .all(|(meta, log_height, univariate_skip)| {
            univariate_skip.skip_rounds <= log_height
                && univariate_skip.zero_check_round_poly.0.len()
                    <= meta.zero_check_uv_degree.saturating_sub(1) << univariate_skip.skip_rounds
                && univariate_skip.eval_check_round_poly.0.len()
                    <= meta.eval_check_uv_degree << univariate_skip.skip_rounds
        })
        || proof.regular_sumcheck.compressed_round_polys.len() != max_regular_rounds
        || !proof
            .regular_sumcheck
            .compressed_round_polys
            .iter()
            .all(|compressed_round_poly| {
                compressed_round_poly.0.len() <= vk.max_regular_sumcheck_degree() + 1
            })
        || proof.regular_sumcheck.evals.len() != vk.metas().len()
        || !izip!(vk.metas(), &proof.regular_sumcheck.evals)
            .all(|(meta, evals)| evals.len() == 2 * meta.width)
        || proof.univariate_eval_sumcheck.compressed_round_polys.len() != max_skip_rounds
        || !proof
            .univariate_eval_sumcheck
            .compressed_round_polys
            .iter()
            .all(|compressed_round_poly| compressed_round_poly.0.len() <= 2)
        || proof.univariate_eval_sumcheck.evals.len() != vk.metas().len()
        || !izip!(
            vk.metas(),
            cloned(&skip_rounds),
            &proof.univariate_eval_sumcheck.evals
        )
        .all(|(meta, skip_rounds, evals)| skip_rounds == 0 || evals.len() == 2 * meta.width)
    {
        return Err(Error::InvalidProofShape);
    }

    let r = (0..max_regular_rounds)
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

    let z = {
        let claims = izip!(
            vk.metas(),
            claims_fs,
            cloned(log_heights),
            &proof.univariate_skips,
            delta.powers()
        )
        .map(
            |(meta, claims_fs, log_height, univariate_skip, delta_power)| {
                let eval_check_claim = random_linear_combine(claims_fs, alpha);
                let claim = if univariate_skip.skip_rounds == 0 {
                    eval_check_claim
                } else {
                    if meta.interaction_count > 0 {
                        let evaluations = evaluations_on_domain(
                            univariate_skip.skip_rounds,
                            &univariate_skip.eval_check_round_poly,
                        );
                        let z_fs_skipped = &z_fs.rslice(log_height)[..univariate_skip.skip_rounds];
                        if evaluate_ml_poly(&evaluations, z_fs_skipped) != eval_check_claim {
                            return Err(Error::UnivariateSkipEvaluationMismatch);
                        }
                    }
                    univariate_skip.zero_check_round_poly.subclaim(x)
                        * (x.exp_power_of_2(univariate_skip.skip_rounds) - Val::ONE)
                        + univariate_skip.eval_check_round_poly.subclaim(x)
                };
                Ok(claim * delta_power)
            },
        )
        .try_collect::<_, Vec<_>, _>()?;

        let claims_at_round = |round| {
            izip!(&regular_rounds, &claims)
                .filter(|(regular_rounds, _)| max_regular_rounds == *regular_rounds + round)
                .map(|(_, subclaim)| *subclaim)
                .sum::<Challenge>()
        };

        let mut claim = claims_at_round(0);
        let z = enumerate(&proof.regular_sumcheck.compressed_round_polys)
            .map(|(round, compressed_round_poly)| {
                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();
                claim = compressed_round_poly.subclaim(claim, z_i) + claims_at_round(round + 1);
                z_i
            })
            .collect_vec();

        let eval = izip!(
            vk.metas(),
            inputs,
            cloned(&skip_rounds),
            cloned(&regular_rounds),
            &proof.regular_sumcheck.evals,
            delta.powers(),
        )
        .map(
            |(meta, input, skip_rounds, regular_rounds, evals, delta_power)| {
                let sels = selectors_at_point(skip_rounds, x);
                let z_fs = z_fs.rslice(regular_rounds);
                let r = r.rslice(regular_rounds);
                let z = z.rslice(regular_rounds);
                let eq_0_z = eq_eval(repeat_n(&Challenge::ZERO, z.len()), z);
                let eq_1_z = eq_eval(repeat_n(&Challenge::ONE, z.len()), z);
                let is_first_row = sels.is_first_row * eq_0_z;
                let is_last_row = sels.is_last_row * eq_1_z;
                let is_transition = Challenge::ONE - eq_1_z + sels.is_transition * eq_1_z;
                let mut builder = VerifierConstraintFolder {
                    main: RowMajorMatrixView::new(evals, meta.width),
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
                    * eq_eval(r, z);
                let eval_check_eval = builder.eval_check_accumulator * eq_eval(z_fs, z);
                (zero_check_eval + eval_check_eval) * delta_power
            },
        )
        .sum::<Challenge>();
        if eval != claim {
            return Err(Error::OodEvaluationMismatch);
        }

        z
    };

    cloned(proof.regular_sumcheck.evals.iter().flatten())
        .for_each(|eval| challenger.observe_algebra_element(eval));

    let eta: Challenge = challenger.sample_algebra_element();
    let theta: Challenge = challenger.sample_algebra_element();

    let z_prime = {
        let claims = izip!(
            cloned(&skip_rounds),
            &proof.regular_sumcheck.evals,
            theta.powers()
        )
        .map(|(skip_rounds, evals, theta_power)| {
            (skip_rounds != 0)
                .then(|| random_linear_combine(evals, eta) * theta_power)
                .unwrap_or_default()
        })
        .collect_vec();

        let claim_at_round = |round| {
            izip!(cloned(&skip_rounds), cloned(&claims))
                .flat_map(|(skip_rounds, subclaim)| {
                    (max_skip_rounds == skip_rounds + round).then_some(subclaim)
                })
                .sum::<Challenge>()
        };

        let mut claim = claim_at_round(0);
        let z_prime = enumerate(&proof.univariate_eval_sumcheck.compressed_round_polys)
            .map(|(round, compressed_round_poly)| {
                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_prime_i = challenger.sample_algebra_element();
                claim =
                    compressed_round_poly.subclaim(claim, z_prime_i) + claim_at_round(round + 1);
                z_prime_i
            })
            .collect_vec();

        let eval = izip!(
            cloned(&skip_rounds),
            &proof.univariate_eval_sumcheck.evals,
            theta.powers()
        )
        .map(|(skip_rounds, evals, theta_power)| {
            random_linear_combine(evals, eta)
                * evaluate_ml_poly(&lagrange_evals(skip_rounds, x), z_prime.rslice(skip_rounds))
                * theta_power
        })
        .sum::<Challenge>();
        if eval != claim {
            return Err(Error::OodEvaluationMismatch);
        }

        z_prime
    };

    cloned(proof.univariate_eval_sumcheck.evals.iter().flatten())
        .for_each(|eval| challenger.observe_algebra_element(eval));

    let zs = izip!(skip_rounds, regular_rounds)
        .map(|(skip_rounds, regular_rounds)| {
            chain![z_prime.rslice(skip_rounds), z.rslice(regular_rounds)]
                .copied()
                .collect()
        })
        .collect();

    Ok(zs)
}
