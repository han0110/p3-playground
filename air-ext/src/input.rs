use alloc::vec::Vec;

use p3_air::BaseAirWithPublicValues;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

#[derive(Clone, Debug)]
pub struct VerifierInput<Val, A> {
    pub air: A,
    pub public_values: Vec<Val>,
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

#[derive(Clone, Debug)]
pub struct ProverInput<Val, A> {
    pub air: A,
    pub public_values: Vec<Val>,
    pub trace: RowMajorMatrix<Val>,
}

impl<Val: Field, A> ProverInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>, trace: RowMajorMatrix<Val>) -> Self
    where
        A: BaseAirWithPublicValues<Val>,
    {
        assert_eq!(air.num_public_values(), public_values.len());
        assert_eq!(air.width(), trace.width());
        Self {
            air,
            public_values,
            trace,
        }
    }

    pub fn air(&self) -> &A {
        &self.air
    }

    pub fn public_values(&self) -> &[Val] {
        &self.public_values
    }

    pub fn to_verifier_input(&self) -> VerifierInput<Val, A>
    where
        A: Clone + BaseAirWithPublicValues<Val>,
    {
        VerifierInput::new(self.air.clone(), self.public_values.clone())
    }
}

impl<Val, A> From<ProverInput<Val, A>> for (VerifierInput<Val, A>, RowMajorMatrix<Val>) {
    fn from(value: ProverInput<Val, A>) -> Self {
        (
            VerifierInput {
                air: value.air,
                public_values: value.public_values,
            },
            value.trace,
        )
    }
}
