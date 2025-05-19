use alloc::vec;
use alloc::vec::Vec;

use itertools::{enumerate, izip, zip_eq};
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

pub fn eq_poly_packed<F: Field, EF: ExtensionField<F>>(z: &[EF]) -> Vec<EF::ExtensionPacking> {
    let log_packing_width = log2_strict_usize(F::Packing::WIDTH);
    let (z_lo, z_hi) = z.split_at(z.len().saturating_sub(log_packing_width));
    let mut eq_z_hi = eq_poly_inner(z_hi, EF::ONE);
    eq_z_hi.resize(F::Packing::WIDTH, EF::ZERO);
    eq_poly(z_lo, EF::ExtensionPacking::from_ext_slice(&eq_z_hi))
}

#[instrument(level = "debug", skip_all, fields(log_b = %z.len()))]
pub fn eq_poly<EF, VarEF>(z: &[EF], scalar: VarEF) -> Vec<VarEF>
where
    EF: Copy + Send + Sync + PrimeCharacteristicRing,
    VarEF: Copy + Send + Sync + Algebra<EF>,
{
    let (z_lo, z_hi) = z.split_at(z.len().div_ceil(2));
    let (eq_lo, eq_hi) = join(
        || eq_poly_inner(z_lo, EF::ONE),
        || eq_poly_inner(z_hi, scalar),
    );

    let mut eq = vec![VarEF::ZERO; 1 << z.len()];
    eq.par_chunks_mut(eq_lo.len())
        .zip(&eq_hi)
        .for_each(|(eq, eq_hi)| izip!(eq, &eq_lo).for_each(|(eq, eq_lo)| *eq = *eq_hi * *eq_lo));
    eq
}

fn eq_poly_inner<EF, VarEF>(z: &[EF], scalar: VarEF) -> Vec<VarEF>
where
    EF: Copy + Send + Sync + PrimeCharacteristicRing,
    VarEF: Copy + Send + Sync + Algebra<EF>,
{
    let mut eq = vec![VarEF::ZERO; 1 << z.len()];
    eq[0] = scalar;
    enumerate(z).for_each(|(i, z_i)| eq_expand(&mut eq[..2 << i], *z_i));
    eq
}

pub fn eq_expand<EF, VarEF>(evals: &mut [VarEF], z_i: EF)
where
    EF: Copy + Send + Sync + PrimeCharacteristicRing,
    VarEF: Copy + Send + Sync + Algebra<EF>,
{
    let (lo, hi) = evals.split_at_mut(evals.len() / 2);
    lo.par_iter_mut().zip(hi).for_each(|(lo, hi)| {
        *hi = *lo * z_i;
        *lo -= *hi;
    });
}

pub fn eq_eval<'a, EF: Field>(
    x: impl IntoIterator<Item = &'a EF>,
    y: impl IntoIterator<Item = &'a EF>,
) -> EF {
    EF::product(zip_eq(x, y).map(|(&x, &y)| (x * y).double() + EF::ONE - x - y))
}
