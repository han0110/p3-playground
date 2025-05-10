use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use core::marker::PhantomData;
use core::mem::swap;

use itertools::{Itertools, chain, cloned, izip};
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
    batch_multiplicative_inverse, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::info_span;

use crate::{
    CompressedRoundPoly, FieldSlice, PackedExtensionValue, RoundPoly, eq_poly_packed, fix_var,
    vec_add,
};

mod eval;
mod regular;
mod univariate_skip;

pub(crate) use eval::*;
pub(crate) use regular::*;
pub(crate) use univariate_skip::*;

impl<Challenge: Field> RoundPoly<Challenge> {
    fn from_evals<Val: Field>(evals: impl IntoIterator<Item = Challenge>) -> Self
    where
        Challenge: ExtensionField<Val>,
    {
        let evals = evals.into_iter().collect_vec();
        Self(
            vander_mat_inv((0..evals.len()).map(Val::from_usize))
                .rows()
                .map(|row| dot_product(cloned(&evals), row))
                .collect(),
        )
    }

    fn mul_by_scaled_eq(&self, scalar: Challenge, r_i: Challenge) -> Self {
        let eq = [Challenge::ONE - r_i, r_i.double() - Challenge::ONE];
        let scaled_eq = eq.map(|coeff| coeff * scalar);
        let mut out = vec![Challenge::ZERO; self.0.len() + 1];
        self.0.iter().enumerate().for_each(|(idx, coeff)| {
            out[idx] += *coeff * scaled_eq[0];
            out[idx + 1] += *coeff * scaled_eq[1];
        });
        Self(out)
    }

    fn into_compressed(mut self) -> CompressedRoundPoly<Challenge> {
        if self.0.len() > 1 {
            self.0.remove(1);
        }
        CompressedRoundPoly(self.0)
    }
}

fn vander_mat_inv<F: Field>(points: impl IntoIterator<Item = F>) -> RowMajorMatrix<F> {
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

pub(crate) struct EqHelper<'a, Val: Field, Challenge: ExtensionField<Val>> {
    pub(crate) evals: Vec<Challenge::ExtensionPacking>,
    r: &'a [Challenge],
    one_minus_r_inv: Vec<Challenge>,
    correcting_factors: Vec<Challenge>,
    _marker: PhantomData<Val>,
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>> EqHelper<'a, Val, Challenge> {
    pub(crate) fn new(r: &'a [Challenge]) -> Self {
        let evals = eq_poly_packed(&r[min(r.len(), 1)..]);
        let one_minus_r_inv =
            batch_multiplicative_inverse(&r.iter().map(|r_i| Challenge::ONE - *r_i).collect_vec());
        let correcting_factors = chain![
            [Challenge::ONE],
            one_minus_r_inv[min(r.len(), 1)..]
                .iter()
                .scan(Challenge::ONE, |product, value| {
                    *product *= *value;
                    Some(*product)
                })
        ]
        .collect();
        Self {
            evals,
            r,
            one_minus_r_inv,
            correcting_factors,
            _marker: PhantomData,
        }
    }

    fn r_i(&self, round: usize) -> Challenge {
        self.r[round]
    }

    fn eval_packed(&self, round: usize, idx: usize) -> Challenge::ExtensionPacking {
        self.evals[idx << round]
    }

    fn eval(&self, round: usize, idx: usize) -> Challenge {
        let len = min(Val::Packing::WIDTH, 1 << self.r.len().saturating_sub(1));
        let step = len >> (self.r.len().saturating_sub(1 + round));
        Challenge::from_basis_coefficients_fn(|j| {
            self.evals[0].as_basis_coefficients_slice()[j].as_slice()[step * idx]
        })
    }

    fn eval_0(&self, round: usize, claim: Challenge, eval_1: Challenge) -> Challenge {
        (claim - self.r[round] * eval_1) * self.one_minus_r_inv[round]
    }

    fn correcting_factor(&self, round: usize) -> Challenge {
        self.correcting_factors[round]
    }
}

pub(crate) enum AirTrace<Val: Field, Challenge: ExtensionField<Val>> {
    Packing(RowMajorMatrix<Val::Packing>),
    ExtensionPacking(RowMajorMatrix<Challenge::ExtensionPacking>),
    Extension(RowMajorMatrix<Challenge>),
}

impl<Val: Field, Challenge: ExtensionField<Val>> Default for AirTrace<Val, Challenge> {
    fn default() -> Self {
        Self::Extension(RowMajorMatrix::new(Vec::new(), 0))
    }
}

impl<Val: Field, Challenge: ExtensionField<Val>> AirTrace<Val, Challenge> {
    pub(crate) fn new(trace: impl Matrix<Val>) -> Self {
        const WINDOW: usize = 2;
        let width = trace.width();
        let height = trace.height();
        let log_height = log2_strict_usize(height);
        let log_packing_width = log2_strict_usize(Val::Packing::WIDTH);
        if log_height > log_packing_width {
            let trace = info_span!("pack trace local and next together").in_scope(|| {
                let len = width * height;
                let packed_len = len >> log_packing_width;
                let packed_height = height >> log_packing_width;
                RowMajorMatrix::new(
                    (0..WINDOW * packed_len)
                        .into_par_iter()
                        .map(|i| {
                            let row = (i / width) / WINDOW;
                            let rot = (i / width) % WINDOW;
                            let col = i % width;
                            // SAFETY: row and column are mod by height and width respectively.
                            unsafe {
                                Val::Packing::from_fn(|j| {
                                    trace.get_unchecked(
                                        (row + rot + j * packed_height) % height,
                                        col,
                                    )
                                })
                            }
                        })
                        .collect(),
                    WINDOW * width,
                )
            });
            Self::Packing(trace)
        } else {
            let len = width * height;
            let trace = RowMajorMatrix::new(
                (0..WINDOW * len)
                    .into_par_iter()
                    .map(|i| {
                        let row = (i / width) / WINDOW;
                        let rot = (i / width) % WINDOW;
                        let col = i % width;
                        // SAFETY: row and column are mod by height and width respectively.
                        Challenge::from(unsafe { trace.get_unchecked((row + rot) % height, col) })
                    })
                    .collect(),
                WINDOW * width,
            );
            Self::Extension(trace)
        }
    }

    pub(crate) fn extension_packing(trace: RowMajorMatrix<Challenge::ExtensionPacking>) -> Self {
        if trace.height() == 1 {
            AirTrace::Extension(unpack_row(&trace.values))
        } else {
            AirTrace::ExtensionPacking(trace)
        }
    }

    pub(crate) fn log_height(&self) -> usize {
        match self {
            Self::Packing(trace) => {
                log2_strict_usize(trace.height()) + log2_strict_usize(Val::Packing::WIDTH)
            }
            Self::ExtensionPacking(trace) => {
                log2_strict_usize(trace.height()) + log2_strict_usize(Val::Packing::WIDTH)
            }
            Self::Extension(trace) => log2_strict_usize(trace.height()),
        }
    }

    pub(crate) fn columnwise_dot_product(&self, scalars: &[Challenge]) -> Vec<Challenge> {
        assert_eq!(1 << self.log_height(), scalars.len());

        macro_rules! columnwise_dot_product_packed {
            ($trace:ident) => {{
                let scalars_packed = {
                    let packed_len = scalars.len() / Val::Packing::WIDTH;
                    (0..packed_len)
                        .into_par_iter()
                        .map(|i| {
                            Challenge::ExtensionPacking::from_basis_coefficients_fn(|j| {
                                Val::Packing::from_fn(|k| {
                                    scalars[i + k * packed_len].as_basis_coefficients_slice()[j]
                                })
                            })
                        })
                        .collect::<Vec<_>>()
                };
                $trace
                    .par_rows()
                    .zip(scalars_packed)
                    .par_fold_reduce(
                        || Challenge::ExtensionPacking::zero_vec($trace.width()),
                        |mut acc, (row, scalar)| {
                            acc.slice_add_assign_scaled_iter(row, scalar);
                            acc
                        },
                        vec_add,
                    )
                    .into_iter()
                    .map(|eval| eval.ext_sum())
                    .collect()
            }};
        }

        match self {
            Self::Packing(trace) => columnwise_dot_product_packed!(trace),
            Self::ExtensionPacking(trace) => columnwise_dot_product_packed!(trace),
            Self::Extension(trace) => trace.columnwise_dot_product(scalars),
        }
    }

    #[must_use]
    fn fix_var(&self, z_i: Challenge) -> Self {
        match self {
            Self::Packing(trace) => Self::extension_packing(fix_var(trace.as_view(), z_i.into())),
            Self::ExtensionPacking(trace) => {
                Self::extension_packing(fix_var(trace.as_view(), z_i.into()))
            }
            Self::Extension(trace) => Self::Extension(fix_var(trace.as_view(), z_i)),
        }
    }

    #[must_use]
    fn fix_lo_scalars(&self, scalars: &[Challenge]) -> Self {
        match self {
            Self::Packing(trace) => {
                let mut fixed = RowMajorMatrix::new(
                    vec![Challenge::ExtensionPacking::ZERO; trace.values.len() / scalars.len()],
                    trace.width(),
                );
                fixed
                    .par_rows_mut()
                    .zip(trace.par_row_chunks(scalars.len()))
                    .for_each(|(acc, trace)| {
                        trace.rows().zip(scalars).for_each(|(row, scalar)| {
                            acc.slice_add_assign_scaled_iter(
                                row,
                                Challenge::ExtensionPacking::from(*scalar),
                            );
                        })
                    });
                Self::extension_packing(fixed)
            }
            _ => unimplemented!(),
        }
    }

    #[must_use]
    fn fix_hi_vars(&self, z: &[Challenge]) -> Self {
        match self {
            Self::Packing(trace) if z.len() >= log2_strict_usize(Val::Packing::WIDTH) => {
                let eq_z_packed = eq_poly_packed(z);
                let fixed = RowMajorMatrix::new(
                    (0..trace.values.len() / eq_z_packed.len())
                        .into_par_iter()
                        .map(|i| {
                            (i..trace.values.len())
                                .into_par_iter()
                                .step_by(trace.values.len() / eq_z_packed.len())
                                .zip(&eq_z_packed)
                                .map(|(idx, scalar)| *scalar * trace.values[idx])
                                .sum::<Challenge::ExtensionPacking>()
                                .ext_sum()
                        })
                        .collect(),
                    trace.width(),
                );
                Self::Extension(fixed)
            }
            _ => unimplemented!(),
        }
    }
}

fn unpack_row<Val: Field, Challenge: ExtensionField<Val>>(
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
