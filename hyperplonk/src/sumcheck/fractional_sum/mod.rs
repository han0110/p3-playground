use alloc::vec;
use alloc::vec::Vec;
use core::iter::successors;
use core::marker::PhantomData;

use itertools::{Itertools, chain, izip};
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
    split_base_and_vector,
};

mod regular;

pub(crate) use regular::*;

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
) -> (Vec<Trace<Val, Challenge>>, Vec<Fraction<Challenge>>)
where
    Val: Field,
    Challenge: ExtensionField<Val>,
{
    let width = 2 * fraction_count;
    let mut layers = successors(Some(input_layer), |input| {
        (input.log_height() > 0).then(|| match input {
            Trace::Packing(input) => {
                let mut output =
                    RowMajorMatrix::new(vec![<_>::ZERO; width * input.height() / 2], width);
                output
                    .par_rows_mut()
                    .zip(input.par_row_chunks(2))
                    .for_each(|(output, input)| next_layer(output, input));
                Trace::extension_packing(output)
            }
            Trace::ExtensionPacking(input) => {
                let mut output =
                    RowMajorMatrix::new(vec![<_>::ZERO; width * input.height() / 2], width);
                output
                    .par_rows_mut()
                    .zip(input.par_row_chunks(2))
                    .for_each(|(output, input)| next_layer(output, input));
                Trace::extension_packing(output)
            }
            Trace::Extension(input) => {
                let mut output =
                    RowMajorMatrix::new(vec![<_>::ZERO; width * input.height() / 2], width);
                output
                    .par_rows_mut()
                    .zip(input.par_row_chunks(2))
                    .for_each(|(output, input)| next_layer(output, input));
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

    let layers = layers
        .into_iter()
        .map(|layer| match layer {
            Trace::Packing(layer) => {
                if layer.height() > 2 {
                    let width = 2 * layer.width();
                    return Trace::Packing(RowMajorMatrix::new(layer.values, width));
                }

                let lo = &layer.row_slice(0).unwrap();
                let hi = &layer.row_slice(1).unwrap();
                Trace::Extension(RowMajorMatrix::new(
                    (0..Val::Packing::WIDTH)
                        .flat_map(|i| {
                            chain![
                                lo.chunks(1 + Challenge::DIMENSION),
                                hi.chunks(1 + Challenge::DIMENSION),
                            ]
                            .flat_map(move |row| {
                                let (numer, denom) =
                                    split_base_and_vector::<_, <Challenge>::ExtensionPacking>(row);
                                [
                                    Challenge::from(numer.as_slice()[i]),
                                    Challenge::from_basis_coefficients_fn(|j| {
                                        denom.as_basis_coefficients_slice()[j].as_slice()[i]
                                    }),
                                ]
                            })
                        })
                        .collect_vec(),
                    4 * fraction_count,
                ))
            }
            Trace::ExtensionPacking(layer) => {
                let width = 2 * layer.width();
                Trace::extension_packing(RowMajorMatrix::new(layer.values, width))
            }
            Trace::Extension(layer) => {
                let width = 2 * layer.width();
                Trace::Extension(RowMajorMatrix::new(layer.values, width))
            }
        })
        .collect();

    (layers, sums)
}

#[inline]
fn next_layer<
    Base: Copy + Send + Sync + PrimeCharacteristicRing,
    VecrorSpace: Copy + Algebra<Base> + BasedVectorSpace<Base>,
>(
    output: &mut [VecrorSpace],
    input: RowMajorMatrixView<Base>,
) {
    let lhs = input.row_slice(0).unwrap();
    let rhs = input.row_slice(1).unwrap();
    let fractions_lhs = lhs.chunks(1 + VecrorSpace::DIMENSION);
    let fractions_rhs = rhs.chunks(1 + VecrorSpace::DIMENSION);
    let fractions_out = output.chunks_mut(2);
    izip!(fractions_out, fractions_lhs, fractions_rhs,).for_each(|(out, lhs, rhs)| {
        let (n_l, d_l) = split_base_and_vector::<_, VecrorSpace>(lhs);
        let (n_r, d_r) = split_base_and_vector::<_, VecrorSpace>(rhs);
        out[0] = *d_r * *n_l + *d_l * *n_r;
        out[1] = *d_r * *d_l;
    })
}
