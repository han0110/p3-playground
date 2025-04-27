use alloc::vec::Vec;
use core::iter::repeat_n;

use itertools::{Itertools, chain, cloned, enumerate, izip};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse, dot_product};
use p3_matrix::dense::RowMajorMatrixView;

use crate::{
    AirMeta, Proof, SymbolicAirBuilder, VerifierFolder, ZeroCheckProof, eq_eval, eq_poly,
    lagrange_evals, selectors_at_point,
};

#[derive(Debug)]
pub enum Error {
    InvalidProofShape,
    OodEvaluationMismatch,
}

#[derive(Clone, Debug)]
pub struct VerifierInput<Val, A> {
    pub(crate) air: A,
    pub(crate) public_values: Vec<Val>,
}

impl<Val: Field, A> VerifierInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>) -> Self
    where
        A: BaseAirWithPublicValues<Val>,
    {
        assert_eq!(air.num_public_values(), public_values.len());
        Self { air, public_values }
    }

    pub fn air(&self) -> &A {
        &self.air
    }

    pub fn public_values(&self) -> &[Val] {
        &self.public_values
    }
}

pub fn verify<Val, Challenge, A>(
    inputs: Vec<VerifierInput<Val, A>>,
    proof: &Proof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<(), Error>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<VerifierFolder<'t, Val, Challenge>>,
{
    assert!(!inputs.is_empty());

    // TODO: Preprocess the meta.
    let metas = inputs
        .iter()
        .map(|input| AirMeta::new(input.air()))
        .collect_vec();

    // TODO: Observe commitment.

    cloned(&proof.log_heights)
        .for_each(|log_height| challenger.observe(Val::from_u8(log_height as u8)));
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    // TODO: Verify LogUp with fractional-sum check.

    let _zs = verify_zero_check(
        &metas,
        &inputs,
        &proof.zero_check,
        &proof.log_heights,
        &mut challenger,
    )?;

    // TODO: PCS verify.

    Ok(())
}

pub fn verify_zero_check<Val, Challenge, A>(
    metas: &[AirMeta],
    inputs: &[VerifierInput<Val, A>],
    proof: &ZeroCheckProof<Challenge>,
    log_heights: &[usize],
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Vec<Challenge>>, Error>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<VerifierFolder<'t, Val, Challenge>>,
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
    let max_regular_ronuds = itertools::max(cloned(&regular_rounds)).unwrap();
    let max_multivariate_degree =
        itertools::max(metas.iter().map(|meta| meta.multivariate_degree)).unwrap();

    if proof.univariate_skips.len() != metas.len()
        || !izip!(metas, cloned(log_heights), cloned(&proof.univariate_skips)).all(
            |(meta, log_height, univariate_skip)| {
                univariate_skip.skip_rounds <= log_height
                    && univariate_skip.round_poly.0.len()
                        <= meta.univariate_degree.saturating_sub(1) << univariate_skip.skip_rounds // TODO: More precise degree bound
            },
        )
        || proof.regular_sumcheck.compressed_round_polys.len() != max_regular_ronuds
        || !proof
            .regular_sumcheck
            .compressed_round_polys
            .iter()
            .all(|compressed_round_poly| compressed_round_poly.0.len() <= max_multivariate_degree)
        || proof.regular_sumcheck.evals.len() != metas.len()
        || !izip!(metas, &proof.regular_sumcheck.evals)
            .all(|(meta, evals)| evals.len() == 2 * meta.width)
        || proof.univariate_eval_sumcheck.compressed_round_polys.len() != max_skip_rounds
        || !proof
            .univariate_eval_sumcheck
            .compressed_round_polys
            .iter()
            .all(|compressed_round_poly| compressed_round_poly.0.len() <= 2)
        || proof.univariate_eval_sumcheck.evals.len() != metas.len()
        || !izip!(
            metas,
            cloned(&skip_rounds),
            &proof.univariate_eval_sumcheck.evals
        )
        .all(|(meta, skip_rounds, evals)| skip_rounds == 0 || evals.len() == 2 * meta.width)
    {
        return Err(Error::InvalidProofShape);
    }

    let r = (0..max_regular_ronuds)
        .map(|_| challenger.sample_algebra_element::<Challenge>())
        .collect_vec();
    let r_inv = batch_multiplicative_inverse(&r);

    let alpha = challenger.sample_algebra_element::<Challenge>();

    proof.univariate_skips.iter().for_each(|univariate_skip| {
        challenger.observe(Val::from_u8(univariate_skip.skip_rounds as u8));
        cloned(&univariate_skip.round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
    });

    let x: Challenge = challenger.sample_algebra_element();

    let beta = challenger.sample_algebra_element::<Challenge>();

    let z = {
        let univariate_skip_subclaims = izip!(&proof.univariate_skips, beta.powers())
            .map(|(univariate_skip, beta_power)| {
                univariate_skip.round_poly.subclaim(x)
                    * (x.exp_power_of_2(univariate_skip.skip_rounds) - Val::ONE)
                    * beta_power
            })
            .collect_vec();

        let mut claim = Challenge::ZERO;
        let z = enumerate(&proof.regular_sumcheck.compressed_round_polys)
            .map(|(round, compressed_round_poly)| {
                claim += izip!(cloned(&regular_rounds), cloned(&univariate_skip_subclaims))
                    .flat_map(|(regular_rounds, subclaim)| {
                        (max_regular_ronuds - round == regular_rounds).then_some(subclaim)
                    })
                    .sum::<Challenge>();
                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();
                claim =
                    compressed_round_poly.eq_weighted_subclaim(claim, z_i, r[round], r_inv[round]);
                z_i
            })
            .collect_vec();

        let eval = izip!(
            metas,
            inputs,
            cloned(&skip_rounds),
            cloned(&regular_rounds),
            &proof.regular_sumcheck.evals,
            beta.powers(),
        )
        .map(
            |(meta, input, skip_rounds, regular_rounds, evals, beta_power)| {
                let sels = selectors_at_point(skip_rounds, x);
                let z = &z[z.len() - regular_rounds..];
                let eq_0_z = eq_eval(repeat_n(&Challenge::ZERO, z.len()), z);
                let eq_1_z = eq_eval(repeat_n(&Challenge::ONE, z.len()), z);
                let is_first_row = sels.is_first_row * eq_0_z;
                let is_last_row = sels.is_last_row * eq_1_z;
                let is_transition = Challenge::ONE - eq_1_z + sels.is_transition * eq_1_z;
                let mut builder = VerifierFolder {
                    main: RowMajorMatrixView::new(evals, meta.width),
                    public_values: input.public_values(),
                    is_first_row,
                    is_last_row,
                    is_transition,
                    alpha,
                    accumulator: Challenge::ZERO,
                };
                input.air().eval(&mut builder);
                builder.accumulator * beta_power
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

    let gamma: Challenge = challenger.sample_algebra_element();
    let theta: Challenge = challenger.sample_algebra_element();

    let z_prime = {
        let univariate_eval_subclaims = izip!(&proof.regular_sumcheck.evals, theta.powers())
            .map(|(evals, theta_power)| {
                evals
                    .iter()
                    .copied()
                    .reduce(|acc, item| acc * gamma + item)
                    .unwrap()
                    * theta_power
            })
            .collect_vec();

        let mut claim = Challenge::ZERO;
        let z_prime = enumerate(&proof.univariate_eval_sumcheck.compressed_round_polys)
            .map(|(round, compressed_round_poly)| {
                claim += izip!(cloned(&skip_rounds), cloned(&univariate_eval_subclaims))
                    .flat_map(|(skip_rounds, subclaim)| {
                        (max_skip_rounds - round == skip_rounds).then_some(subclaim)
                    })
                    .sum::<Challenge>();
                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_prime_i = challenger.sample_algebra_element();
                claim = compressed_round_poly.subclaim(claim, z_prime_i);
                z_prime_i
            })
            .collect_vec();

        let eval = izip!(
            cloned(&skip_rounds),
            &proof.univariate_eval_sumcheck.evals,
            theta.powers()
        )
        .map(|(skip_rounds, evals, theta_power)| {
            evals
                .iter()
                .copied()
                .reduce(|acc, item| acc * gamma + item)
                .unwrap_or_default()
                * dot_product::<Challenge, _, _>(
                    eq_poly(&z_prime[z_prime.len() - skip_rounds..], Challenge::ONE).into_iter(),
                    lagrange_evals(skip_rounds, x).into_iter(),
                )
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
            chain![
                &z_prime[z_prime.len() - skip_rounds..],
                &z[z.len() - regular_rounds..]
            ]
            .copied()
            .collect()
        })
        .collect();

    Ok(zs)
}
