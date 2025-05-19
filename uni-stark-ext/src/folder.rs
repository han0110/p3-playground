use alloc::vec::Vec;

use itertools::izip;
use p3_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder};
use p3_field::{BasedVectorSpace, PackedField};
use p3_matrix::dense::RowMajorMatrixView;

use crate::{
    InteractionBuilder, InteractionType, PackedChallenge, PackedVal, StarkGenericConfig, Val,
    ViewPair,
};

#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    pub public_values: &'a Vec<Val<SC>>,
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub alpha_powers: &'a [SC::Challenge],
    pub decomposed_alpha_powers: &'a [&'a [Val<SC>]],
    pub accumulator: PackedChallenge<SC>,
    pub constraint_index: usize,
    pub beta_powers: &'a [PackedChallenge<SC>],
    pub gamma_powers: &'a [PackedChallenge<SC>],
    pub numers: Vec<PackedVal<SC>>,
    pub denoms: Vec<PackedChallenge<SC>>,
    pub interaction_index: usize,
}

#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    pub main: ViewPair<'a, SC::Challenge>,
    pub public_values: &'a Vec<Val<SC>>,
    pub is_first_row: SC::Challenge,
    pub is_last_row: SC::Challenge,
    pub is_transition: SC::Challenge,
    pub alpha: SC::Challenge,
    pub accumulator: SC::Challenge,
    pub beta_powers: &'a [SC::Challenge],
    pub gamma_powers: &'a [SC::Challenge],
    pub numers: Vec<SC::Challenge>,
    pub denoms: Vec<SC::Challenge>,
    pub interaction_index: usize,
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrixView<'a, PackedVal<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: PackedVal<SC> = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedChallenge<SC>>::into(alpha_power) * x;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array: [Self::Expr; N] = array.map(Into::into);
        self.accumulator += PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
            let alpha_powers = &self.decomposed_alpha_powers[i]
                [self.constraint_index..(self.constraint_index + N)];
            PackedVal::<SC>::packed_linear_combination::<N>(alpha_powers, &expr_array)
        });
        self.constraint_index += N;
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for ProverConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for ProverConstraintFolder<'_, SC> {
    type EF = SC::Challenge;

    type ExprEF = PackedChallenge<SC>;

    type VarEF = PackedChallenge<SC>;

    #[inline]
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += x * alpha_power;
        self.constraint_index += 1;
    }
}

impl<SC: StarkGenericConfig> InteractionBuilder for ProverConstraintFolder<'_, SC> {
    const ONLY_INTERACTION: bool = false;

    #[inline]
    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let mut count = count.into();
        if interaction_type == InteractionType::Receive {
            count = -count;
        }
        self.numers[self.interaction_index] = count;

        let mut fields = fields.into_iter();
        self.denoms[self.interaction_index] =
            self.gamma_powers[bus_index] + fields.next().unwrap().into();
        izip!(fields, self.beta_powers).for_each(|(field, beta_power)| {
            self.denoms[self.interaction_index] += *beta_power * field.into();
        });

        self.interaction_index += 1;
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: SC::Challenge = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for VerifierConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for VerifierConstraintFolder<'_, SC> {
    type EF = SC::Challenge;

    type ExprEF = SC::Challenge;

    type VarEF = SC::Challenge;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: SC::Challenge = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<SC: StarkGenericConfig> InteractionBuilder for VerifierConstraintFolder<'_, SC> {
    const ONLY_INTERACTION: bool = false;

    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let mut count = count.into();
        if interaction_type == InteractionType::Receive {
            count = -count;
        }
        self.numers[self.interaction_index] = count;

        let mut fields = fields.into_iter();
        self.denoms[self.interaction_index] =
            self.gamma_powers[bus_index] + fields.next().unwrap().into();
        izip!(fields, self.beta_powers).for_each(|(field, beta_power)| {
            self.denoms[self.interaction_index] += *beta_power * field.into();
        });

        self.interaction_index += 1;
    }
}
