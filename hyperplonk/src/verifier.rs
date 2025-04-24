use alloc::vec::Vec;

use itertools::{Itertools, cloned, izip};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrixView;

use crate::{AirMeta, BatchSumcheckProof, Proof, SymbolicAirBuilder, VerifierFolder};

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
    Val: Field,
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

    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    // TODO: Verify LogUp with fractional sumchecks.

    let _z = verify_zero_sumcheck(&metas, &inputs, &proof.zero_sumcheck, &mut challenger)?;

    // TODO: PCS verify.

    Ok(())
}

pub fn verify_zero_sumcheck<Val, Challenge, A>(
    metas: &[AirMeta],
    inputs: &[VerifierInput<Val, A>],
    proof: &BatchSumcheckProof<Challenge>,
    mut challenger: impl FieldChallenger<Val>,
) -> Result<Vec<Challenge>, Error>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<VerifierFolder<'t, Val, Challenge>>,
{
    let max_degree = itertools::max(metas.iter().map(|meta| meta.degree)).unwrap();
    let max_num_vars = itertools::max(cloned(&proof.num_vars)).unwrap();

    if proof.compressed_round_polys.len() != max_num_vars
        || !proof
            .compressed_round_polys
            .iter()
            .all(|round_poly| round_poly.0.len() == max_degree)
        || !izip!(metas, &proof.evals).all(|(meta, evals)| evals.len() == 2 * meta.width)
    {
        return Err(Error::InvalidProofShape);
    }

    proof
        .num_vars
        .iter()
        .for_each(|num_vars| challenger.observe(Val::from_u8(*num_vars as u8)));

    let r = (0..max_num_vars)
        .map(|_| challenger.sample_algebra_element::<Challenge>())
        .collect_vec();
    let r_inv = batch_multiplicative_inverse(&r);

    let alpha = challenger.sample_algebra_element::<Challenge>();
    let beta = challenger.sample_algebra_element::<Challenge>();

    let mut claim = Challenge::ZERO;
    let z = proof
        .compressed_round_polys
        .iter()
        .enumerate()
        .map(|(round, compressed_round_poly)| {
            compressed_round_poly
                .0
                .iter()
                .for_each(|coeff| challenger.observe_algebra_element(*coeff));
            let z_i = challenger.sample_algebra_element();
            claim = compressed_round_poly.subclaim(claim, z_i, r[round], r_inv[round]);
            z_i
        })
        .collect_vec();

    let sum = izip!(inputs, metas, &proof.num_vars, &proof.evals)
        .map(|(input, meta, num_vars, evals)| {
            let z = &z[z.len() - *num_vars..];
            let is_first_row = Challenge::product(z.iter().map(|z_i| Challenge::ONE - *z_i));
            let is_last_row = Challenge::product(z.iter().copied());
            let is_transition = Challenge::ONE - is_last_row;
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
            builder.accumulator
        })
        .reduce(|acc, value| acc * beta + value)
        .unwrap();
    (sum == claim)
        .then_some(z)
        .ok_or(Error::OodEvaluationMismatch)
}
