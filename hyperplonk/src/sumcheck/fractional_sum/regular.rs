use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::{Algebra, BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;

use crate::{
    CompressedRoundPoly, EqHelper, PackedExtensionValue, RingArray, RoundPoly, Trace, eq_eval,
    split_base_and_vector,
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

    pub(crate) fn compute_round_poly(&mut self, round: usize) -> CompressedRoundPoly<Challenge> {
        let round_poly = match &self.trace {
            Trace::Packing(_) => self.compute_eq_weighted_round_poly_packing(round),
            Trace::ExtensionPacking(_) => {
                self.compute_eq_weighted_round_poly_extension_packing(round)
            }
            Trace::Extension(_) => self.compute_eq_weighted_round_poly_extension(round),
        };

        self.round_poly = round_poly.clone();

        round_poly
            .mul_by_scaled_eq(self.eq_eval, self.eq_helper.z_i(round))
            .into_compressed()
    }

    fn compute_eq_weighted_round_poly_packing(&mut self, round: usize) -> RoundPoly<Challenge> {
        let Trace::Packing(trace) = &self.trace else {
            unreachable!()
        };

        let RingArray([coeff_0, coeff_2]) = trace
            .par_row_chunks(2)
            .enumerate()
            .map(|(row, chunk)| self.eval(chunk) * self.eq_helper.eval_packed(round, row))
            .sum();

        self.recover_eq_weighted_round_poly(round, coeff_0.ext_sum(), coeff_2.ext_sum())
    }

    fn compute_eq_weighted_round_poly_extension_packing(
        &mut self,
        round: usize,
    ) -> RoundPoly<Challenge> {
        let Trace::ExtensionPacking(trace) = &self.trace else {
            unreachable!()
        };

        let RingArray([coeff_0, coeff_2]) = trace
            .par_row_chunks(2)
            .enumerate()
            .map(|(row, chunk)| self.eval(chunk) * self.eq_helper.eval_packed(round, row))
            .sum();

        self.recover_eq_weighted_round_poly(round, coeff_0.ext_sum(), coeff_2.ext_sum())
    }

    fn compute_eq_weighted_round_poly_extension(&mut self, round: usize) -> RoundPoly<Challenge> {
        let Trace::Extension(trace) = &self.trace else {
            unreachable!()
        };

        let RingArray([coeff_0, coeff_2]) = trace
            .par_row_chunks(2)
            .enumerate()
            .map(|(row, chunk)| self.eval(chunk) * self.eq_helper.eval(round, row))
            .sum();

        self.recover_eq_weighted_round_poly(round, coeff_0, coeff_2)
    }

    fn eval<Var, VarEF>(&self, chunk: RowMajorMatrixView<Var>) -> RingArray<VarEF, 2>
    where
        Var: Copy + Send + Sync + PrimeCharacteristicRing,
        VarEF: Copy + Algebra<Var> + From<Challenge> + BasedVectorSpace<Var>,
    {
        let lo = chunk.row_slice(0).unwrap();
        let hi = chunk.row_slice(1).unwrap();
        let (lhs_lo, rhs_lo) = lo.split_at(lo.len() / 2);
        let (lhs_hi, rhs_hi) = hi.split_at(hi.len() / 2);
        izip!(
            lhs_lo.chunks(1 + VarEF::DIMENSION),
            rhs_lo.chunks(1 + VarEF::DIMENSION),
            lhs_hi.chunks(1 + VarEF::DIMENSION),
            rhs_hi.chunks(1 + VarEF::DIMENSION),
            self.alpha_powers.chunks(2)
        )
        .map(|(lhs_lo, rhs_lo, lhs_hi, rhs_hi, alpha_powers)| {
            let (&n_l_lo, &d_l_lo) = split_base_and_vector::<_, VarEF>(lhs_lo);
            let (&n_r_lo, &d_r_lo) = split_base_and_vector::<_, VarEF>(rhs_lo);
            let (&n_l_hi, &d_l_hi) = split_base_and_vector::<_, VarEF>(lhs_hi);
            let (&n_r_hi, &d_r_hi) = split_base_and_vector::<_, VarEF>(rhs_hi);
            let alpha_power_even = VarEF::from(alpha_powers[0]);
            let alpha_power_odd = VarEF::from(alpha_powers[1]);
            RingArray([
                alpha_power_even * (d_r_lo * n_l_lo + d_l_lo * n_r_lo)
                    + alpha_power_odd * (d_l_lo * d_r_lo),
                alpha_power_even
                    * ((d_r_hi - d_r_lo) * (n_l_hi - n_l_lo)
                        + (d_l_hi - d_l_lo) * (n_r_hi - n_r_lo))
                    + alpha_power_odd * ((d_l_hi - d_l_lo) * (d_r_hi - d_r_lo)),
            ])
        })
        .sum::<RingArray<_, 2>>()
    }

    fn recover_eq_weighted_round_poly(
        &self,
        round: usize,
        mut coeff_0: Challenge,
        mut coeff_2: Challenge,
    ) -> RoundPoly<Challenge> {
        coeff_0 *= self.eq_helper.correcting_factor(round);
        coeff_2 *= self.eq_helper.correcting_factor(round);
        let eval_1 = self.eq_helper.recover_eval_1(round, self.claim, coeff_0);
        let coeff_1 = eval_1 - coeff_0 - coeff_2;
        RoundPoly(vec![coeff_0, coeff_1, coeff_2])
    }

    pub(crate) fn fix_var(&mut self, round: usize, z_i: Challenge) {
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
        self.eq_eval *= eq_eval([&self.eq_helper.z_i(round)], [&z_i]);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        self.trace.into_evals()
    }
}
