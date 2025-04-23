use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrixView;

use crate::{AirMeta, Proof, SumcheckProof, SymbolicAirBuilder, VerifierFolder};

#[derive(Debug)]
pub enum Error {
    InvalidProofShape,
    OodEvaluationMismatch,
}

pub fn verify<Val, Challenge, A>(
    air: &A,
    public_values: &[Val],
    proof: &Proof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<(), Error>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<VerifierFolder<'t, Val, Challenge>>,
{
    // TODO: Preprocess the meta.
    let meta = AirMeta::new(air);

    // TODO: Observe commitment.

    challenger.observe_slice(public_values);

    // TODO: Verify LogUp with fractional sumchecks.

    let _z = verify_zero_sumcheck(
        air,
        meta.degree,
        public_values,
        &proof.zero_sumcheck,
        challenger,
    )?;

    // TODO: PCS verify.

    Ok(())
}

pub fn verify_zero_sumcheck<Val, Challenge, A>(
    air: &A,
    degree: usize,
    public_values: &[Val],
    proof: &SumcheckProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Challenge>, Error>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<VerifierFolder<'t, Val, Challenge>>,
{
    let width = air.width();
    let num_vars = proof.num_vars;

    if proof.compressed_round_polys.len() != proof.num_vars
        || !proof
            .compressed_round_polys
            .iter()
            .all(|round_poly| round_poly.0.len() == degree)
        || proof.evals.len() != 2 * air.width()
    {
        return Err(Error::InvalidProofShape);
    }

    if num_vars == 0 {
        return Ok(Vec::new());
    }

    challenger.observe(Val::from_u8(num_vars as u8));

    let r = (0..num_vars)
        .map(|_| challenger.sample_algebra_element::<Challenge>())
        .collect_vec();
    let alpha = challenger.sample_algebra_element::<Challenge>();

    let r_inv = batch_multiplicative_inverse(&r);

    let mut claim = Challenge::ZERO;
    let mut z = Vec::with_capacity(num_vars);
    proof
        .compressed_round_polys
        .iter()
        .enumerate()
        .for_each(|(round, compressed_round_poly)| {
            compressed_round_poly
                .0
                .iter()
                .for_each(|coeff| challenger.observe_algebra_element(*coeff));
            let z_i = challenger.sample_algebra_element();
            claim = compressed_round_poly.subclaim(claim, z_i, r[round], r_inv[round]);
            z.push(z_i);
        });

    let is_first_row = Challenge::product(z.iter().map(|z_i| Challenge::ONE - *z_i));
    let is_last_row = Challenge::product(z.iter().copied());
    let is_transition = Challenge::ONE - is_last_row;
    let mut builder = VerifierFolder {
        main: RowMajorMatrixView::new(&proof.evals, width),
        public_values,
        is_first_row,
        is_last_row,
        is_transition,
        alpha,
        accumulator: Challenge::ZERO,
    };
    air.eval(&mut builder);
    if builder.accumulator != claim {
        return Err(Error::OodEvaluationMismatch);
    }

    Ok(z)
}
