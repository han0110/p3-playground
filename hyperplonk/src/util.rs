use alloc::vec;
use alloc::vec::Vec;

use itertools::{enumerate, izip, rev, zip_eq};
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

pub(crate) fn horner<'a, F: Field>(
    coeffs: impl IntoIterator<IntoIter: DoubleEndedIterator, Item = &'a F>,
    x: F,
) -> F {
    rev(coeffs.into_iter().copied())
        .reduce(|acc, coeff| acc * x + coeff)
        .unwrap_or_default()
}

pub(crate) fn eq_poly_packed<F: Field, E: ExtensionField<F>>(r: &[E]) -> Vec<E::ExtensionPacking> {
    let log_packing_width = log2_strict_usize(F::Packing::WIDTH);
    let (r_lo, r_hi) = r.split_at(r.len().saturating_sub(log_packing_width));
    let mut eq_r_hi = eq_poly(r_hi, E::ONE);
    eq_r_hi.resize(F::Packing::WIDTH, E::ZERO);
    eq_poly(r_lo, E::ExtensionPacking::from_ext_slice(&eq_r_hi))
}

#[instrument(level = "debug", skip_all, fields(log_h = %r.len()))]
pub fn eq_poly<F: Field, VarEF: Copy + Send + Sync + Algebra<F>>(
    r: &[F],
    scalar: VarEF,
) -> Vec<VarEF> {
    let mut evals = vec![VarEF::ZERO; 1 << r.len()];
    evals[0] = scalar;
    enumerate(r).for_each(|(i, r_i)| eq_expand(&mut evals, *r_i, i));
    evals
}

pub(crate) fn eq_expand<F: Field, VarEF: Copy + Send + Sync + Algebra<F>>(
    evals: &mut [VarEF],
    x_i: F,
    i: usize,
) {
    let (lo, hi) = evals[..2 << i].split_at_mut(1 << i);
    lo.par_iter_mut().zip(hi).for_each(|(lo, hi)| {
        *hi = *lo * x_i;
        *lo -= *hi;
    });
}

pub fn eq_eval<'a, F: Field>(
    x: impl IntoIterator<Item = &'a F>,
    y: impl IntoIterator<Item = &'a F>,
) -> F {
    F::product(zip_eq(x, y).map(|(&x, &y)| (x * y).double() + F::ONE - x - y))
}

#[instrument(level = "debug", skip_all, fields(log_h = %mat.height().ilog2()))]
pub(crate) fn fix_var<F, EF>(mat: RowMajorMatrixView<F>, z_i: EF) -> RowMajorMatrix<EF>
where
    F: Copy + Send + Sync + PrimeCharacteristicRing,
    EF: Copy + Send + Sync + Algebra<F>,
{
    RowMajorMatrix::new(
        mat.par_row_chunks(2)
            .flat_map(|rows| {
                rows.values[..mat.width()]
                    .into_par_iter()
                    .zip(&rows.values[mat.width()..])
                    .map(|(lo, hi)| z_i * (*hi - *lo) + *lo)
            })
            .collect(),
        mat.width(),
    )
}

pub(crate) trait PackedExtensionValue<F: Field, E: ExtensionField<F, ExtensionPacking = Self>>:
    PackedFieldExtension<F, E> + Sync + Send
{
    #[inline]
    fn ext_sum(&self) -> E {
        E::from_basis_coefficients_fn(|i| {
            self.as_basis_coefficients_slice()[i]
                .as_slice()
                .iter()
                .copied()
                .sum()
        })
    }
}

impl<F, E, T> PackedExtensionValue<F, E> for T
where
    F: Field,
    E: ExtensionField<F, ExtensionPacking = T>,
    T: PackedFieldExtension<F, E> + Sync + Send,
{
}

pub trait FieldSlice<F: Copy + PrimeCharacteristicRing>: AsMut<[F]> {
    #[inline]
    fn slice_assign_iter(&mut self, rhs: impl IntoIterator<Item = F>) {
        izip!(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs = rhs);
    }

    #[inline]
    fn slice_scale(&mut self, scalar: F) {
        self.as_mut().iter_mut().for_each(|lhs| *lhs *= scalar);
    }

    #[inline]
    fn slice_add_assign(&mut self, rhs: &[F]) {
        izip!(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs += *rhs);
    }

    #[inline]
    fn slice_sub_iter(
        &mut self,
        lhs: impl IntoIterator<Item = F>,
        rhs: impl IntoIterator<Item = F>,
    ) {
        izip!(self.as_mut(), lhs, rhs).for_each(|(out, lhs, rhs): (&mut F, _, _)| *out = lhs - rhs);
    }

    #[inline]
    fn slice_sub_assign(&mut self, rhs: &[F]) {
        izip!(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs -= *rhs);
    }

    #[inline]
    fn slice_add_assign_scaled_iter<R, S>(&mut self, rhs: impl IntoIterator<Item = R>, scalar: S)
    where
        F: Algebra<S>,
        S: Copy + Algebra<R>,
    {
        izip!(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs += scalar * rhs);
    }

    #[inline]
    fn slice_sub_assign_scaled_iter<R, S>(&mut self, rhs: impl IntoIterator<Item = R>, scalar: S)
    where
        F: Algebra<S>,
        S: Copy + Algebra<R>,
    {
        izip!(self.as_mut(), rhs).for_each(|(lhs, rhs)| *lhs -= scalar * rhs);
    }
}

impl<F: Copy + PrimeCharacteristicRing> FieldSlice<F> for [F] {}

#[inline]
pub(crate) fn vec_add<F: Copy + PrimeCharacteristicRing>(mut lhs: Vec<F>, rhs: Vec<F>) -> Vec<F> {
    lhs.slice_add_assign(&rhs);
    lhs
}
