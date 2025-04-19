use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::*;

use p3_air::{AirBuilder, AirBuilderWithPublicValues};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;

#[derive(Debug)]
pub struct ProverFolderWithVal<'a, F, EF, Var, VarEF> {
    pub main: RowMajorMatrixView<'a, Var>,
    pub public_values: &'a [F],
    pub is_first_row: Var,
    pub is_last_row: Var,
    pub is_transition: Var,
    pub alpha_powers: &'a [EF],
    pub accumulator: VarEF,
    pub constraint_index: usize,
}

impl<'a, F, EF, Var, VarEF> AirBuilder for ProverFolderWithVal<'a, F, EF, Var, VarEF>
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
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += VarEF::from(alpha_power) * x;
        self.constraint_index += 1;
    }
}

impl<F, EF, Var, VarEF> AirBuilderWithPublicValues for ProverFolderWithVal<'_, F, EF, Var, VarEF>
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

// FIXME: Figure a way to have a single `ProverFolder` but support `main` being
//        matrix with base field values or extension field packed values.
//        The main constraint is `AirBuilder` requires `Var: Algebra<F>` but
//        `EF::ExtensionPacking` only has `Algebra<EF>`.
#[derive(Debug)]
pub struct ProverFolderWithExtensionPacking<'a, F: Field, EF: ExtensionField<F>> {
    pub main: RowMajorMatrixView<'a, ExtensionPacking<F, EF>>,
    pub public_values: &'a [F],
    pub is_first_row: ExtensionPacking<F, EF>,
    pub is_last_row: ExtensionPacking<F, EF>,
    pub is_transition: ExtensionPacking<F, EF>,
    pub alpha_powers: &'a [EF],
    pub accumulator: ExtensionPacking<F, EF>,
    pub constraint_index: usize,
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder
    for ProverFolderWithExtensionPacking<'a, F, EF>
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
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += ExtensionPacking(EF::ExtensionPacking::from(alpha_power)) * x;
        self.constraint_index += 1;
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues
    for ProverFolderWithExtensionPacking<'_, F, EF>
{
    type PublicVar = F;

    #[inline]
    fn public_values(&self) -> &[F] {
        self.public_values
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
pub struct VerifierFolder<'a, F, EF> {
    pub main: RowMajorMatrixView<'a, EF>,
    pub public_values: &'a [F],
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub alpha: EF,
    pub accumulator: EF,
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder for VerifierFolder<'a, F, EF> {
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
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues for VerifierFolder<'_, F, EF> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}
