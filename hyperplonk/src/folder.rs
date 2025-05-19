use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::*;

use itertools::izip;
use p3_air::{AirBuilder, AirBuilderWithPublicValues};
use p3_air_ext::{InteractionBuilder, InteractionType};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;

pub type ProverConstraintFolderOnExtension<'t, Val, Challenge> =
    ProverConstraintFolderGeneric<'t, Val, Challenge, Challenge, Challenge>;

pub type ProverConstraintFolderOnPacking<'t, Val, Challenge> = ProverConstraintFolderGeneric<
    't,
    Val,
    Challenge,
    <Val as Field>::Packing,
    <Challenge as ExtensionField<Val>>::ExtensionPacking,
>;

#[derive(Debug)]
pub struct ProverConstraintFolderGeneric<'a, F, EF, Var, VarEF> {
    pub main: RowMajorMatrixView<'a, Var>,
    pub public_values: &'a [F],
    pub is_first_row: Var,
    pub is_last_row: Var,
    pub is_transition: Var,
    pub beta_powers: &'a [EF],
    pub gamma_powers: &'a [EF],
    pub zero_check_alpha_powers: &'a [EF],
    pub eval_check_alpha_powers: &'a [EF],
    pub zero_check_accumulator: VarEF,
    pub eval_check_accumulator: VarEF,
    pub constraint_index: usize,
    pub interaction_index: usize,
}

impl<'a, F, EF, Var, VarEF> AirBuilder for ProverConstraintFolderGeneric<'a, F, EF, Var, VarEF>
where
    F: Field,
    EF: ExtensionField<F>,
    Var: Algebra<F> + Copy + Send + Sync,
    VarEF: Algebra<Var> + From<EF>,
{
    type F = F;
    type Expr = Var;
    type Var = Var;
    type M = RowMajorMatrixView<'a, Var>;

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
            panic!("only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        let alpha_power = self.zero_check_alpha_powers[self.constraint_index];
        self.zero_check_accumulator += VarEF::from(alpha_power) * x;
        self.constraint_index += 1;
    }
}

impl<F, EF, Var, VarEF> AirBuilderWithPublicValues
    for ProverConstraintFolderGeneric<'_, F, EF, Var, VarEF>
where
    F: Field,
    EF: ExtensionField<F>,
    Var: Algebra<F> + Copy + Send + Sync,
    VarEF: Algebra<Var> + From<EF>,
{
    type PublicVar = F;

    #[inline]
    fn public_values(&self) -> &[F] {
        self.public_values
    }
}

impl<F, EF, Var, VarEF> InteractionBuilder for ProverConstraintFolderGeneric<'_, F, EF, Var, VarEF>
where
    F: Field,
    EF: ExtensionField<F>,
    Var: Algebra<F> + Copy + Send + Sync,
    VarEF: Algebra<Var> + From<EF>,
{
    const ONLY_INTERACTION: bool = false;

    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let mut numer = count.into();
        if interaction_type == InteractionType::Receive {
            numer = -numer;
        }
        let alpha_power = self.eval_check_alpha_powers[2 * self.interaction_index];
        self.eval_check_accumulator += VarEF::from(alpha_power) * numer;

        let denom = {
            let mut fields = fields.into_iter();
            VarEF::from(self.gamma_powers[bus_index])
                + fields.next().unwrap().into()
                + izip!(fields, self.beta_powers)
                    .map(|(field, beta_power)| VarEF::from(*beta_power) * field.into())
                    .sum::<VarEF>()
        };
        let alpha_power = self.eval_check_alpha_powers[2 * self.interaction_index + 1];
        self.eval_check_accumulator += VarEF::from(alpha_power) * denom;

        self.interaction_index += 1;
    }
}

// FIXME: Figure a way to have a single `ProverConstraintFolder` but support `main` being
//        matrix with base field values or extension field packed values.
//        The main constraint is `AirBuilder` requires `Var: Algebra<F>` but
//        `EF::ExtensionPacking` only has `Algebra<EF>`.
#[derive(Debug)]
pub struct ProverConstraintFolderOnExtensionPacking<'a, F: Field, EF: ExtensionField<F>> {
    pub main: RowMajorMatrixView<'a, ExtensionPacking<F, EF>>,
    pub public_values: &'a [F],
    pub is_first_row: ExtensionPacking<F, EF>,
    pub is_last_row: ExtensionPacking<F, EF>,
    pub is_transition: ExtensionPacking<F, EF>,
    pub beta_powers: &'a [EF],
    pub gamma_powers: &'a [EF],
    pub zero_check_alpha_powers: &'a [EF],
    pub eval_check_alpha_powers: &'a [EF],
    pub zero_check_accumulator: ExtensionPacking<F, EF>,
    pub eval_check_accumulator: ExtensionPacking<F, EF>,
    pub constraint_index: usize,
    pub interaction_index: usize,
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder
    for ProverConstraintFolderOnExtensionPacking<'a, F, EF>
{
    type F = F;
    type Expr = ExtensionPacking<F, EF>;
    type Var = ExtensionPacking<F, EF>;
    type M = RowMajorMatrixView<'a, ExtensionPacking<F, EF>>;

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
            panic!("only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        let alpha_power = self.zero_check_alpha_powers[self.constraint_index];
        self.zero_check_accumulator.0 += x.0 * alpha_power;
        self.constraint_index += 1;
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues
    for ProverConstraintFolderOnExtensionPacking<'_, F, EF>
{
    type PublicVar = F;

    #[inline]
    fn public_values(&self) -> &[F] {
        self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> InteractionBuilder
    for ProverConstraintFolderOnExtensionPacking<'_, F, EF>
{
    const ONLY_INTERACTION: bool = false;

    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let mut numer = count.into();
        if interaction_type == InteractionType::Receive {
            numer = -numer;
        }
        let alpha_power = self.eval_check_alpha_powers[2 * self.interaction_index];
        self.eval_check_accumulator.0 += numer.0 * alpha_power;

        let denom = {
            let mut fields = fields.into_iter();
            fields.next().unwrap().into().0
                + izip!(fields, self.beta_powers)
                    .map(|(field, beta_power)| field.into().0 * *beta_power)
                    .sum::<EF::ExtensionPacking>()
                + self.gamma_powers[bus_index]
        };
        let alpha_power = self.eval_check_alpha_powers[2 * self.interaction_index + 1];
        self.eval_check_accumulator.0 += denom * alpha_power;

        self.interaction_index += 1;
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct ExtensionPacking<F: Field, EF: ExtensionField<F>>(pub EF::ExtensionPacking);

impl<F: Field, EF: ExtensionField<F>> ExtensionPacking<F, EF> {
    #[inline]
    pub fn from_slice(values: &[EF::ExtensionPacking]) -> &[Self] {
        // SAFETY: repr(transparent) ensures transmutation safety.
        unsafe { transmute(values) }
    }
}

impl<F: Field, EF: ExtensionField<F>> PrimeCharacteristicRing for ExtensionPacking<F, EF> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self(EF::ExtensionPacking::ZERO);
    const ONE: Self = Self(EF::ExtensionPacking::ONE);
    const TWO: Self = Self(EF::ExtensionPacking::TWO);
    const NEG_ONE: Self = Self(EF::ExtensionPacking::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self(EF::ExtensionPacking::from(F::Packing::from(
            F::from_prime_subfield(f),
        )))
    }
}

impl<F: Field, EF: ExtensionField<F>> Algebra<F> for ExtensionPacking<F, EF> {}

impl<F: Field, EF: ExtensionField<F>> From<F> for ExtensionPacking<F, EF> {
    #[inline]
    fn from(value: F) -> Self {
        Self(EF::ExtensionPacking::from(F::Packing::from(value)))
    }
}

impl<F: Field, EF: ExtensionField<F>> Neg for ExtensionPacking<F, EF> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<F: Field, EF: ExtensionField<F>> Add for ExtensionPacking<F, EF> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<F: Field, EF: ExtensionField<F>> Add<F> for ExtensionPacking<F, EF> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self::Output {
        Self(self.0 + F::Packing::from(rhs))
    }
}

impl<F: Field, EF: ExtensionField<F>> AddAssign for ExtensionPacking<F, EF> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<F: Field, EF: ExtensionField<F>> AddAssign<F> for ExtensionPacking<F, EF> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        self.0 += F::Packing::from(rhs);
    }
}

impl<F: Field, EF: ExtensionField<F>> Sub for ExtensionPacking<F, EF> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<F: Field, EF: ExtensionField<F>> Sub<F> for ExtensionPacking<F, EF> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self::Output {
        Self(self.0 - F::Packing::from(rhs))
    }
}

impl<F: Field, EF: ExtensionField<F>> SubAssign for ExtensionPacking<F, EF> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<F: Field, EF: ExtensionField<F>> SubAssign<F> for ExtensionPacking<F, EF> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        self.0 -= F::Packing::from(rhs);
    }
}

impl<F: Field, EF: ExtensionField<F>> Mul for ExtensionPacking<F, EF> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<F: Field, EF: ExtensionField<F>> Mul<F> for ExtensionPacking<F, EF> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0 * F::Packing::from(rhs))
    }
}

impl<F: Field, EF: ExtensionField<F>> MulAssign for ExtensionPacking<F, EF> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<F: Field, EF: ExtensionField<F>> MulAssign<F> for ExtensionPacking<F, EF> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        self.0 *= F::Packing::from(rhs);
    }
}

impl<F: Field, EF: ExtensionField<F>> Sum for ExtensionPacking<F, EF> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::ZERO)
    }
}

impl<F: Field, EF: ExtensionField<F>> Product for ExtensionPacking<F, EF> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::ONE)
    }
}

#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, F, EF> {
    pub main: RowMajorMatrixView<'a, EF>,
    pub public_values: &'a [F],
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub alpha: EF,
    pub beta_powers: &'a [EF],
    pub gamma_powers: &'a [EF],
    pub zero_check_accumulator: EF,
    pub eval_check_accumulator: EF,
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder for VerifierConstraintFolder<'a, F, EF> {
    type F = F;
    type Expr = EF;
    type Var = EF;
    type M = RowMajorMatrixView<'a, EF>;

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
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: EF = x.into();
        self.zero_check_accumulator *= self.alpha;
        self.zero_check_accumulator += x;
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues
    for VerifierConstraintFolder<'_, F, EF>
{
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> InteractionBuilder for VerifierConstraintFolder<'_, F, EF> {
    const ONLY_INTERACTION: bool = false;

    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let mut numer = count.into();
        if interaction_type == InteractionType::Receive {
            numer = -numer;
        }
        self.eval_check_accumulator *= self.alpha;
        self.eval_check_accumulator += numer;

        let denom = {
            let mut fields = fields.into_iter();
            self.gamma_powers[bus_index]
                + fields.next().unwrap().into()
                + izip!(fields, self.beta_powers)
                    .map(|(field, beta_power)| *beta_power * field.into())
                    .sum::<EF>()
        };
        self.eval_check_accumulator *= self.alpha;
        self.eval_check_accumulator += denom;
    }
}
