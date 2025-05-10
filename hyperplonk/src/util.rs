use alloc::vec;
use alloc::vec::Vec;

use itertools::{cloned, enumerate, rev, zip_eq};
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

pub(crate) fn random_linear_combine<'a, EF: Field>(
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

pub fn evaluate_ml_poly<F: Field, EF: ExtensionField<F>>(evals: &[F], z: &[EF]) -> EF {
    match z {
        [] => EF::from(evals[0]),
        [z_0] => *z_0 * (evals[1] - evals[0]) + evals[0],
        &[ref z @ .., z_i] => {
            let (lo, hi) = evals.split_at(evals.len() / 2);
            let (lo, hi) = join(|| evaluate_ml_poly(lo, z), || evaluate_ml_poly(hi, z));
            z_i * (hi - lo) + lo
        }
    }
}

pub(crate) fn eq_poly_packed<F: Field, EF: ExtensionField<F>>(
    r: &[EF],
) -> Vec<EF::ExtensionPacking> {
    let log_packing_width = log2_strict_usize(F::Packing::WIDTH);
    let (r_lo, r_hi) = r.split_at(r.len().saturating_sub(log_packing_width));
    let mut eq_r_hi = eq_poly(r_hi, EF::ONE);
    eq_r_hi.resize(F::Packing::WIDTH, EF::ZERO);
    eq_poly(r_lo, EF::ExtensionPacking::from_ext_slice(&eq_r_hi))
}

#[instrument(level = "debug", skip_all, fields(log_h = %r.len()))]
pub fn eq_poly<EF, VarEF>(r: &[EF], scalar: VarEF) -> Vec<VarEF>
where
    EF: Field,
    VarEF: Copy + Send + Sync + Algebra<EF>,
{
    let mut evals = vec![VarEF::ZERO; 1 << r.len()];
    evals[0] = scalar;
    enumerate(r).for_each(|(i, r_i)| eq_expand(&mut evals, *r_i, i));
    evals
}

pub(crate) fn eq_expand<EF, VarEF>(evals: &mut [VarEF], x_i: EF, i: usize)
where
    EF: Field,
    VarEF: Copy + Send + Sync + Algebra<EF>,
{
    let (lo, hi) = evals[..2 << i].split_at_mut(1 << i);
    lo.par_iter_mut().zip(hi).for_each(|(lo, hi)| {
        *hi = *lo * x_i;
        *lo -= *hi;
    });
}

pub fn eq_eval<'a, EF: Field>(
    x: impl IntoIterator<Item = &'a EF>,
    y: impl IntoIterator<Item = &'a EF>,
) -> EF {
    EF::product(zip_eq(x, y).map(|(&x, &y)| (x * y).double() + EF::ONE - x - y))
}

#[instrument(level = "debug", skip_all, fields(log_h = %mat.height().ilog2()))]
pub(crate) fn fix_var<Var, VarEF>(mat: RowMajorMatrixView<Var>, z_i: VarEF) -> RowMajorMatrix<VarEF>
where
    Var: Copy + Send + Sync + PrimeCharacteristicRing,
    VarEF: Copy + Send + Sync + Algebra<Var>,
{
    RowMajorMatrix::new(
        mat.par_row_chunks(2)
            .flat_map(|rows| {
                let (lo, hi) = rows.values.split_at(mat.width());
                lo.into_par_iter()
                    .zip(hi)
                    .map(|(lo, hi)| z_i * (*hi - *lo) + *lo)
            })
            .collect(),
        mat.width(),
    )
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
