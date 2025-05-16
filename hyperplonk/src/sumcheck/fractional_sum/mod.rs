use alloc::vec;
use alloc::vec::Vec;
use core::iter::successors;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_air::Air;
use p3_air_ext::VerifierInput;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::{
    AirMeta, Fraction, ProverInteractionFolderOnExtension, ProverInteractionFolderOnPacking, Trace,
    random_linear_combine, split_base_and_vector,
};

mod regular;

pub(crate) use regular::*;

pub(crate) const FS_LOG_ARITY: usize = 1;
pub(crate) const FS_ARITY: usize = 1 << FS_LOG_ARITY;

#[instrument(skip_all)]
pub(crate) fn fractional_sum_trace<Val, Challenge, A>(
    meta: &AirMeta,
    input: &VerifierInput<Val, A>,
    trace: &Trace<Val, Challenge>,
    beta_powers: &[Challenge],
    gamma_powers: &[Challenge],
) -> Option<Trace<Val, Challenge>>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverInteractionFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverInteractionFolderOnExtension<'t, Val, Challenge>>,
{
    (meta.interaction_count != 0).then(|| match trace {
        Trace::Packing(trace) => {
            let width = (1 + Challenge::DIMENSION) * meta.interaction_count;
            let mut input_layer =
                RowMajorMatrix::new(vec![Val::Packing::ZERO; width * trace.height()], width);
            input_layer
                .par_rows_mut()
                .zip(trace.par_row_slices())
                .for_each(|(fractions, row)| {
                    let mut builder = ProverInteractionFolderOnPacking {
                        main: RowMajorMatrixView::new(row, meta.width),
                        public_values: input.public_values(),
                        beta_powers,
                        gamma_powers,
                        fractions,
                        _marker: PhantomData,
                    };
                    input.air().eval(&mut builder);
                });
            Trace::Packing(input_layer)
        }
        Trace::Extension(trace) => {
            let width = 2 * meta.interaction_count;
            let mut input_layer =
                RowMajorMatrix::new(vec![Challenge::ZERO; width * trace.height()], width);
            input_layer
                .par_rows_mut()
                .zip(trace.par_row_slices())
                .for_each(|(fractions, row)| {
                    let mut builder = ProverInteractionFolderOnExtension {
                        main: RowMajorMatrixView::new(row, meta.width),
                        public_values: input.public_values(),
                        beta_powers,
                        gamma_powers,
                        fractions,
                        _marker: PhantomData,
                    };
                    input.air().eval(&mut builder);
                });
            Trace::Extension(input_layer)
        }
        _ => unreachable!(),
    })
}

#[instrument(skip_all)]
pub(crate) fn fractional_sum_layers<Val, Challenge>(
    fraction_count: usize,
    input_layer: Trace<Val, Challenge>,
) -> (Vec<Fraction<Challenge>>, Vec<Trace<Val, Challenge>>)
where
    Val: Field,
    Challenge: ExtensionField<Val>,
{
    let width = 2 * fraction_count;
    let mut layers = successors(Some(input_layer), |input| {
        (input.log_b() > 0).then(|| match input {
            Trace::Packing(input) => {
                let mut output =
                    RowMajorMatrix::new(vec![<_>::ZERO; width * input.height() / FS_ARITY], width);
                output
                    .par_rows_mut()
                    .zip(input.par_row_chunks(FS_ARITY))
                    .for_each(|(output, input)| {
                        eval_fractional_sum_row(FS_ARITY, input.values, output)
                    });
                Trace::extension_packing(output)
            }
            Trace::ExtensionPacking(input) => {
                let mut output =
                    RowMajorMatrix::new(vec![<_>::ZERO; width * input.height() / FS_ARITY], width);
                output
                    .par_rows_mut()
                    .zip(input.par_row_chunks(FS_ARITY))
                    .for_each(|(output, input)| {
                        eval_fractional_sum_row(FS_ARITY, input.values, output)
                    });
                Trace::extension_packing(output)
            }
            Trace::Extension(input) => {
                let mut output =
                    RowMajorMatrix::new(vec![<_>::ZERO; width * input.height() / FS_ARITY], width);
                output
                    .par_rows_mut()
                    .zip(input.par_row_chunks(FS_ARITY))
                    .for_each(|(output, input)| {
                        eval_fractional_sum_row(FS_ARITY, input.values, output)
                    });
                Trace::Extension(output)
            }
        })
    })
    .collect_vec();

    let sums = {
        let Some(Trace::Extension(output_layer)) = layers.pop() else {
            unreachable!()
        };
        output_layer
            .values
            .into_iter()
            .tuples()
            .map(|(numer, denom)| Fraction { numer, denom })
            .collect()
    };

    let width = 2 * fraction_count * FS_ARITY;
    let layers = layers
        .into_iter()
        .map(|layer| {
            let layer = match layer {
                Trace::Packing(layer) if layer.height() > FS_ARITY => {
                    let width = (1 + Challenge::DIMENSION) * fraction_count * FS_ARITY;
                    Trace::Packing(RowMajorMatrix::new(layer.values, width))
                }
                Trace::Packing(layer) => Trace::Extension(RowMajorMatrix::new(
                    (0..Val::Packing::WIDTH)
                        .flat_map(|i| {
                            layer.values.chunks(1 + Challenge::DIMENSION).flat_map(
                                move |fraction| {
                                    let (numer, denom): (_, &Challenge::ExtensionPacking) =
                                        split_base_and_vector(fraction);
                                    [
                                        Challenge::from(numer.as_slice()[i]),
                                        Challenge::from_basis_coefficients_fn(|j| {
                                            denom.as_basis_coefficients_slice()[j].as_slice()[i]
                                        }),
                                    ]
                                },
                            )
                        })
                        .collect_vec(),
                    width,
                )),
                Trace::ExtensionPacking(layer) => {
                    Trace::extension_packing(RowMajorMatrix::new(layer.values, width))
                }
                Trace::Extension(layer) => {
                    Trace::Extension(RowMajorMatrix::new(layer.values, width))
                }
            };
            layer
        })
        .collect();

    (sums, layers)
}

#[inline]
fn eval_fractional_sum_row<
    Base: Copy + Send + Sync + PrimeCharacteristicRing,
    VecrorSpace: Copy + Algebra<Base> + BasedVectorSpace<Base>,
>(
    arity: usize,
    input: &[Base],
    output: &mut [VecrorSpace],
) {
    eval_fractional_sum(arity, input, output, |output, (numer, denom)| {
        output[0] = numer;
        output[1] = denom;
        &mut output[2..]
    });
}

#[inline]
pub(crate) fn eval_fractional_sum_accumulator<Base, VecrorSpace>(
    arity: usize,
    input: &[Base],
    alpha: VecrorSpace,
) -> VecrorSpace
where
    Base: Copy + Send + Sync + PrimeCharacteristicRing,
    VecrorSpace: Copy + Algebra<Base> + BasedVectorSpace<Base>,
{
    eval_fractional_sum(arity, input, VecrorSpace::ZERO, |acc, (numer, denom)| {
        random_linear_combine(&[acc, numer, denom], alpha)
    })
}

#[inline]
fn eval_fractional_sum<Base, VecrorSpace, T>(
    arity: usize,
    input: &[Base],
    fold_init: T,
    mut fold: impl FnMut(T, (VecrorSpace, VecrorSpace)) -> T,
) -> T
where
    Base: Copy + Send + Sync + PrimeCharacteristicRing,
    VecrorSpace: Copy + Algebra<Base> + BasedVectorSpace<Base>,
{
    let fraction_size = 1 + VecrorSpace::DIMENSION;
    let chunk_size = input.len() / arity;
    match arity {
        2 => izip!(
            input[..chunk_size].chunks(fraction_size),
            input[chunk_size..].chunks(fraction_size),
        )
        .fold(fold_init, |acc, (f_0, f_1)| {
            let (n_0, d_0) = split_base_and_vector::<_, VecrorSpace>(f_0);
            let (n_1, d_1) = split_base_and_vector::<_, VecrorSpace>(f_1);
            fold(acc, fractional_sum([n_0, n_1], [d_0, d_1]))
        }),
        _ => unimplemented!(),
    }
}

#[inline]
fn fractional_sum<Var: Copy, VarEF: Copy + Algebra<Var>, const N: usize>(
    n: [&Var; N],
    d: [&VarEF; N],
) -> (VarEF, VarEF) {
    match N {
        2 => (*d[1] * *n[0] + *d[0] * *n[1], *d[0] * *d[1]),
        _ => unimplemented!(),
    }
}

#[inline]
pub(crate) fn fractional_sum_at_zero_and_inf<
    Var: Copy + PrimeCharacteristicRing,
    VarEF: Copy + Algebra<Var>,
    const N: usize,
>(
    n_lo: [&Var; N],
    d_lo: [&VarEF; N],
    n_hi: [&Var; N],
    d_hi: [&VarEF; N],
) -> (VarEF, VarEF, VarEF, VarEF) {
    match N {
        2 => (
            *d_lo[1] * *n_lo[0] + *d_lo[0] * *n_lo[1],
            *d_lo[0] * *d_lo[1],
            (*d_hi[1] - *d_lo[1]) * (*n_hi[0] - *n_lo[0])
                + (*d_hi[0] - *d_lo[0]) * (*n_hi[1] - *n_lo[1]),
            (*d_hi[0] - *d_lo[0]) * (*d_hi[1] - *d_lo[1]),
        ),
        _ => unimplemented!(),
    }
}
