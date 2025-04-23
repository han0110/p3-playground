use alloc::vec;
use alloc::vec::Vec;
use core::mem::swap;

use itertools::{Itertools, enumerate, izip, zip_eq};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

#[inline]
pub(crate) fn vec_add<F: Copy + PrimeCharacteristicRing>(mut lhs: Vec<F>, rhs: Vec<F>) -> Vec<F> {
    lhs.slice_add_assign(&rhs);
    lhs
}

pub(crate) fn horner<'a, F: Field>(
    coeffs: impl IntoIterator<IntoIter: DoubleEndedIterator, Item = &'a F>,
    x: F,
) -> F {
    coeffs
        .into_iter()
        .copied()
        .rev()
        .reduce(|acc, coeff| acc * x + coeff)
        .unwrap_or_default()
}

pub(crate) fn vander_mat_inv<F: Field>(points: impl IntoIterator<Item = F>) -> RowMajorMatrix<F> {
    let points = points.into_iter().map_into().collect_vec();
    assert!(!points.is_empty());

    let poly_from_roots = |poly: &mut [F], roots: &[F], scalar: F| {
        *poly.last_mut().unwrap() = scalar;
        izip!(2.., roots).for_each(|(len, root)| {
            let mut tmp = scalar;
            (0..poly.len() - 1).rev().take(len).for_each(|idx| {
                tmp = poly[idx] - tmp * *root;
                swap(&mut tmp, &mut poly[idx])
            })
        });
    };

    let mut mat = RowMajorMatrix::new(F::zero_vec(points.len() * points.len()), points.len());
    izip!(mat.rows_mut(), 0.., &points).for_each(|(col, j, point_j)| {
        let point_is = izip!(0.., &points)
            .filter(|(i, _)| *i != j)
            .map(|(_, point_i)| *point_i)
            .collect_vec();
        let scalar = F::product(point_is.iter().map(|point_i| *point_j - *point_i)).inverse();
        poly_from_roots(col, &point_is, scalar);
    });
    mat.transpose()
}

#[instrument(level = "debug", skip_all, fields(dim = %r.len()))]
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

#[instrument(level = "debug", skip_all, fields(dim = %mat.height().ilog2()))]
pub(crate) fn fix_var<
    F: Copy + Send + Sync + PrimeCharacteristicRing,
    EF: Copy + Send + Sync + Algebra<F>,
>(
    mat: RowMajorMatrixView<F>,
    z_i: EF,
) -> RowMajorMatrix<EF> {
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

pub(crate) fn unpack_row<Val: Field, Challenge: ExtensionField<Val>>(
    row: &[Challenge::ExtensionPacking],
) -> RowMajorMatrix<Challenge> {
    let width = row.len();
    RowMajorMatrix::new(
        (0..width * Val::Packing::WIDTH)
            .into_par_iter()
            .map(|i| {
                Challenge::from_basis_coefficients_fn(|j| {
                    row[i % width].as_basis_coefficients_slice()[j].as_slice()[i / width]
                })
            })
            .collect(),
        width,
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
