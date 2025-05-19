use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::eq_poly;

/// A (not necessarily hiding) multilinear polynomial commitment scheme, for committing to (batches of) polynomials
pub trait MlPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Self::Val>,
{
    /// The element in matrix to be committed.
    type Val: Field;

    /// The commitment that's sent to the verifier.
    type Commitment: Clone + Debug + Serialize + DeserializeOwned;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData;

    /// Type of the output of `get_evaluations`.
    type Evaluations<'a>: Matrix<Self::Val> + 'a;

    /// The opening argument.
    type Proof: Clone + Debug + Serialize + DeserializeOwned;

    type Error: Debug;

    fn commit(
        &self,
        evaluations: Vec<RowMajorMatrix<Self::Val>>,
    ) -> (Self::Commitment, Self::ProverData);

    fn get_evaluations<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
    ) -> Self::Evaluations<'a>;

    /// Open rounds of matrices with queries given the corresponding
    /// evaluations.
    ///
    /// The caller should be responsible to observe the evaluations.
    #[allow(clippy::type_complexity)]
    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // for each query:
                Vec<(
                    // the query,
                    MlQuery<Challenge>,
                    // values at the query
                    Vec<Challenge>,
                )>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> Self::Proof;

    /// Verify opening of rounds of matrices with queries given the
    /// corresponding evaluations.
    ///
    /// The caller should be responsible to observe the evaluations.
    #[allow(clippy::type_complexity)]
    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<
                // for each query:
                Vec<(
                    // the query,
                    MlQuery<Challenge>,
                    // values at the query
                    Vec<Challenge>,
                )>,
            >,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug)]
pub enum MlQuery<Challenge> {
    Eq(Vec<Challenge>),
    EqRotateRight(Vec<Challenge>, usize),
}

impl<Challenge: Field> MlQuery<Challenge> {
    pub fn log_b(&self) -> usize {
        match self {
            Self::Eq(z) | Self::EqRotateRight(z, _) => z.len(),
        }
    }

    pub fn to_mle(&self, scalar: Challenge) -> Vec<Challenge> {
        match self {
            Self::Eq(z) => eq_poly(z, scalar),
            Self::EqRotateRight(z, mid) => {
                let mut mle = eq_poly(z, scalar);
                mle.rotate_right(*mid);
                mle
            }
        }
    }
}
