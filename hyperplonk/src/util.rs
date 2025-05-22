use alloc::vec;
use alloc::vec::Vec;
use core::array::from_fn;
use core::iter::Sum;
use core::ops::{Add, Mul};

use itertools::{cloned, izip, rev, zip_eq};
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

pub(crate) fn random_linear_combine<'a, EF: 'a + Copy + PrimeCharacteristicRing>(
    values: impl IntoIterator<Item = &'a EF>,
    r: EF,
) -> EF {
    cloned(values).fold(EF::ZERO, |acc, value| acc * r + value)
}

pub(crate) fn evaluate_uv_poly<'a, F: Field, EF: ExtensionField<F>>(
    coeffs: impl IntoIterator<IntoIter: DoubleEndedIterator, Item = &'a F>,
    x: EF,
) -> EF {
    rev(cloned(coeffs)).fold(EF::ZERO, |acc, coeff| acc * x + coeff)
}

pub(crate) fn evaluate_ml_poly<Var, VarEF>(evals: &[Var], z: &[VarEF]) -> VarEF
where
    Var: Copy + Send + Sync + PrimeCharacteristicRing,
    VarEF: Copy + Send + Sync + Algebra<Var>,
{
    match z {
        [] => VarEF::from(evals[0]),
        [z_0] => *z_0 * (evals[1] - evals[0]) + evals[0],
        &[ref z @ .., z_i] => {
            let (lo, hi) = evals.split_at(evals.len() / 2);
            let (lo, hi) = join(|| evaluate_ml_poly(lo, z), || evaluate_ml_poly(hi, z));
            z_i * (hi - lo) + lo
        }
    }
}

#[instrument(level = "debug", skip_all, fields(log_b = %mat.height().ilog2()))]
pub(crate) fn fix_var<Var, VarEF>(mat: RowMajorMatrixView<Var>, z_i: VarEF) -> RowMajorMatrix<VarEF>
where
    Var: Copy + Send + Sync + PrimeCharacteristicRing,
    VarEF: Copy + Send + Sync + Algebra<Var>,
{
    let mut fixed = RowMajorMatrix::new(vec![VarEF::ZERO; mat.values.len() / 2], mat.width());
    fixed
        .par_rows_mut()
        .zip(mat.par_row_chunks(2))
        .for_each(|(out, rows)| {
            izip!(out, rows.row(0).unwrap(), rows.row(1).unwrap())
                .for_each(|(out, lo, hi)| *out = z_i * (hi - lo) + lo)
        });
    fixed
}

#[instrument(level = "debug", skip_all, fields(log_b = %mat.height().ilog2()))]
pub(crate) fn fix_var_extension_packing<Var, VarEF>(
    mat: RowMajorMatrixView<VarEF>,
    z_i: Var,
) -> RowMajorMatrix<VarEF>
where
    Var: Copy + Send + Sync,
    VarEF: Copy + Send + Sync + Algebra<Var>,
{
    let mut fixed = RowMajorMatrix::new(vec![VarEF::ZERO; mat.values.len() / 2], mat.width());
    fixed
        .par_rows_mut()
        .zip(mat.par_row_chunks(2))
        .for_each(|(out, rows)| {
            izip!(out, rows.row(0).unwrap(), rows.row(1).unwrap())
                .for_each(|(out, lo, hi)| *out = (hi - lo) * z_i + lo)
        });
    fixed
}

pub(crate) trait PackedExtensionValue<F: Field, EF: ExtensionField<F, ExtensionPacking = Self>>:
    PackedFieldExtension<F, EF> + Sync + Send
{
    #[inline]
    fn ext_sum(&self) -> EF {
        EF::from_basis_coefficients_fn(|i| {
            self.as_basis_coefficients_slice()[i]
                .as_slice()
                .iter()
                .copied()
                .sum()
        })
    }
}

impl<F, EF, T> PackedExtensionValue<F, EF> for T
where
    F: Field,
    EF: ExtensionField<F, ExtensionPacking = T>,
    T: PackedFieldExtension<F, EF> + Sync + Send,
{
}

pub trait RSlice<T> {
    fn rslice(&self, len: usize) -> &[T];
}

impl<T> RSlice<T> for [T] {
    fn rslice(&self, len: usize) -> &[T] {
        &self[self.len() - len..]
    }
}

pub trait FieldSlice<F: Copy + PrimeCharacteristicRing>: AsMut<[F]> {
    #[inline]
    fn slice_assign_iter(&mut self, rhs: impl IntoIterator<Item = F>) {
        zip_eq(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs = rhs);
    }

    #[inline]
    fn slice_scale(&mut self, scalar: F) {
        self.as_mut().iter_mut().for_each(|lhs| *lhs *= scalar);
    }

    #[inline]
    fn slice_add_assign(&mut self, rhs: &[F]) {
        zip_eq(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs += *rhs);
    }

    #[inline]
    fn slice_sub_iter(
        &mut self,
        lhs: impl IntoIterator<Item = F>,
        rhs: impl IntoIterator<Item = F>,
    ) {
        zip_eq(self.as_mut(), zip_eq(lhs, rhs))
            .for_each(|(out, (lhs, rhs)): (&mut _, _)| *out = lhs - rhs);
    }

    #[inline]
    fn slice_sub_assign(&mut self, rhs: &[F]) {
        zip_eq(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs -= *rhs);
    }

    #[inline]
    fn slice_add_assign_scaled_iter<R, S>(&mut self, rhs: impl IntoIterator<Item = R>, scalar: S)
    where
        F: Algebra<S>,
        S: Copy + Algebra<R>,
    {
        zip_eq(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs += scalar * rhs);
    }

    #[inline]
    fn slice_sub_assign_scaled_iter<R, S>(&mut self, rhs: impl IntoIterator<Item = R>, scalar: S)
    where
        F: Algebra<S>,
        S: Copy + Algebra<R>,
    {
        zip_eq(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs -= scalar * rhs);
    }
}

impl<F: Copy + PrimeCharacteristicRing> FieldSlice<F> for [F] {}

#[inline]
pub(crate) fn vec_add<F: Copy + PrimeCharacteristicRing>(mut lhs: Vec<F>, rhs: Vec<F>) -> Vec<F> {
    lhs.slice_add_assign(&rhs);
    lhs
}

#[inline]
pub(crate) fn vec_pair_add<F: Copy + PrimeCharacteristicRing>(
    mut lhs: (Vec<F>, Vec<F>),
    rhs: (Vec<F>, Vec<F>),
) -> (Vec<F>, Vec<F>) {
    lhs.0.slice_add_assign(&rhs.0);
    lhs.1.slice_add_assign(&rhs.1);
    lhs
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct RingArray<F: Copy + PrimeCharacteristicRing, const N: usize>(pub [F; N]);

impl<F: Copy + PrimeCharacteristicRing, const N: usize> Default for RingArray<F, N> {
    #[inline]
    fn default() -> Self {
        Self([F::ZERO; N])
    }
}

impl<F: Copy + PrimeCharacteristicRing, const N: usize> Add for RingArray<F, N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<F: Copy + PrimeCharacteristicRing, const N: usize> Mul<F> for RingArray<F, N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0.map(|lhs| lhs * rhs))
    }
}

impl<F: Copy + PrimeCharacteristicRing, const N: usize> Sum for RingArray<F, N> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item).unwrap_or_default()
    }
}
