use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::{Algebra, BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;

use crate::{
    CompressedRoundPoly, EqHelper, FS_ARITY, PackedExtensionValue, RingArray, RoundPoly, Trace,
    eq_eval, fractional_sum_at_zero_and_inf, split_base_and_vector,
};

pub(crate) struct FractionalSumRegularProver<'a, Val: Field, Challenge: ExtensionField<Val>> {
    fraction_count: usize,
    claim: Challenge,
    trace: Trace<Val, Challenge>,
    alpha_powers: &'a [Challenge],
    eq_helper: EqHelper<'a, Val, Challenge>,
    round_poly: RoundPoly<Challenge>,
    eq_eval: Challenge,
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>>
    FractionalSumRegularProver<'a, Val, Challenge>
{
    pub(crate) fn new(
        fraction_count: usize,
        claim: Challenge,
        trace: Trace<Val, Challenge>,
        alpha_powers: &'a [Challenge],
        z: &'a [Challenge],
    ) -> Self {
        let eq_helper = EqHelper::new(z);
        Self {
            fraction_count,
            claim,
            trace,
            alpha_powers,
            eq_helper,
            round_poly: Default::default(),
            eq_eval: Challenge::ONE,
        }
    }

    pub(crate) fn compute_round_poly(&mut self, log_b: usize) -> CompressedRoundPoly<Challenge> {
        let round_poly = match &self.trace {
            Trace::Packing(_) => self.compute_eq_weighted_round_poly_packing(log_b),
            Trace::ExtensionPacking(_) => {
                self.compute_eq_weighted_round_poly_extension_packing(log_b)
            }
            Trace::Extension(_) => self.compute_eq_weighted_round_poly_extension(log_b),
        };

        self.round_poly = round_poly.clone();

        round_poly
            .mul_by_scaled_eq(self.eq_eval, self.eq_helper.z_i(log_b))
            .into_compressed()
    }

    fn compute_eq_weighted_round_poly_packing(&mut self, log_b: usize) -> RoundPoly<Challenge> {
        let Trace::Packing(trace) = &self.trace else {
            unreachable!()
        };

        let RingArray([coeff_0, coeff_2]) = trace
            .par_row_chunks(2)
            .enumerate()
            .map(|(row, chunk)| self.eval(chunk) * self.eq_helper.eval_packed(log_b, row))
            .sum();

        self.recover_eq_weighted_round_poly(log_b, coeff_0.ext_sum(), coeff_2.ext_sum())
    }

    fn compute_eq_weighted_round_poly_extension_packing(
        &mut self,
        log_b: usize,
    ) -> RoundPoly<Challenge> {
        let Trace::ExtensionPacking(trace) = &self.trace else {
            unreachable!()
        };

        let RingArray([coeff_0, coeff_2]) = trace
            .par_row_chunks(2)
            .enumerate()
            .map(|(row, chunk)| self.eval(chunk) * self.eq_helper.eval_packed(log_b, row))
            .sum();

        self.recover_eq_weighted_round_poly(log_b, coeff_0.ext_sum(), coeff_2.ext_sum())
    }

    fn compute_eq_weighted_round_poly_extension(&mut self, log_b: usize) -> RoundPoly<Challenge> {
        let Trace::Extension(trace) = &self.trace else {
            unreachable!()
        };

        let RingArray([coeff_0, coeff_2]) = trace
            .par_row_chunks(2)
            .enumerate()
            .map(|(row, chunk)| self.eval(chunk) * self.eq_helper.eval(log_b, row))
            .sum();

        self.recover_eq_weighted_round_poly(log_b, coeff_0, coeff_2)
    }

    fn eval<Var, VarEF>(&self, chunk: RowMajorMatrixView<Var>) -> RingArray<VarEF, 2>
    where
        Var: Copy + Send + Sync + PrimeCharacteristicRing,
        VarEF: Copy + Algebra<Var> + From<Challenge> + BasedVectorSpace<Var>,
    {
        let lo = chunk.row_slice(0).unwrap();
        let hi = chunk.row_slice(1).unwrap();
        let chunk_size = lo.len() / FS_ARITY;
        izip!(
            lo[..chunk_size].chunks(1 + VarEF::DIMENSION),
            lo[chunk_size..].chunks(1 + VarEF::DIMENSION),
            hi[..chunk_size].chunks(1 + VarEF::DIMENSION),
            hi[chunk_size..].chunks(1 + VarEF::DIMENSION),
            self.alpha_powers.chunks(2)
        )
        .map(|(f_lo_0, f_lo_1, f_hi_0, f_hi_1, alpha_powers)| {
            let (n_0_lo, d_0_lo) = split_base_and_vector::<_, VarEF>(f_lo_0);
            let (n_1_lo, d_1_lo) = split_base_and_vector::<_, VarEF>(f_lo_1);
            let (n_0_hi, d_0_hi) = split_base_and_vector::<_, VarEF>(f_hi_0);
            let (n_1_hi, d_1_hi) = split_base_and_vector::<_, VarEF>(f_hi_1);
            let (n_0, d_0, n_inf, d_inf) = fractional_sum_at_zero_and_inf(
                [n_0_lo, n_1_lo],
                [d_0_lo, d_1_lo],
                [n_0_hi, n_1_hi],
                [d_0_hi, d_1_hi],
            );
            let alpha_power_even = VarEF::from(alpha_powers[0]);
            let alpha_power_odd = VarEF::from(alpha_powers[1]);
            RingArray([
                alpha_power_even * n_0 + alpha_power_odd * d_0,
                alpha_power_even * n_inf + alpha_power_odd * d_inf,
            ])
        })
        .sum::<RingArray<_, 2>>()
    }

    fn recover_eq_weighted_round_poly(
        &self,
        log_b: usize,
        mut coeff_0: Challenge,
        mut coeff_2: Challenge,
    ) -> RoundPoly<Challenge> {
        coeff_0 *= self.eq_helper.correcting_factor(log_b);
        coeff_2 *= self.eq_helper.correcting_factor(log_b);
        let eval_1 = self.eq_helper.recover_eval_1(log_b, self.claim, coeff_0);
        let coeff_1 = eval_1 - coeff_0 - coeff_2;
        RoundPoly(vec![coeff_0, coeff_1, coeff_2])
    }

    pub(crate) fn fix_var(&mut self, log_b: usize, z_i: Challenge) {
        self.trace = match &self.trace {
            Trace::Packing(trace) => {
                let trace = RowMajorMatrix::new(
                    trace
                        .par_row_chunks(2)
                        .flat_map(|rows| {
                            let (lo, hi) = rows.values.split_at(trace.width());
                            lo.par_chunks(1 + Challenge::DIMENSION)
                                .zip(hi.par_chunks(1 + Challenge::DIMENSION))
                                .flat_map(|(lo, hi)| {
                                    let (numer_lo, denom_lo) =
                                        split_base_and_vector::<_, Challenge::ExtensionPacking>(lo);
                                    let (numer_hi, denom_hi) =
                                        split_base_and_vector::<_, Challenge::ExtensionPacking>(hi);
                                    [
                                        Challenge::ExtensionPacking::from(z_i)
                                            * (*numer_hi - *numer_lo)
                                            + *numer_lo,
                                        Challenge::ExtensionPacking::from(z_i)
                                            * (*denom_hi - *denom_lo)
                                            + *denom_lo,
                                    ]
                                })
                        })
                        .collect(),
                    4 * self.fraction_count,
                );
                Trace::extension_packing(trace)
            }
            _ => self.trace.fix_var(z_i),
        };
        self.claim = self.round_poly.subclaim(z_i);
        self.eq_eval *= eq_eval([&self.eq_helper.z_i(log_b)], [&z_i]);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        self.trace.into_evals()
    }
}
