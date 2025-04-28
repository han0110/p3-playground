use alloc::vec;
use alloc::vec::Vec;

use itertools::cloned;
use p3_field::{Field, FieldArray, dot_product};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{CompressedRoundPoly, RoundPoly, fix_var};

pub(crate) struct EvalSumcheckProver<'a, Challenge> {
    pub(crate) claim: Challenge,
    pub(crate) trace: RowMajorMatrix<Challenge>,
    pub(crate) weight: Vec<Challenge>,
    pub(crate) gamma_powers: &'a [Challenge],
    pub(crate) starting_round: usize,
    pub(crate) round_poly: RoundPoly<Challenge>,
}

impl<Challenge: Field> EvalSumcheckProver<'_, Challenge> {
    #[instrument(skip_all, name = "compute eval round poly", fields(log_h = %self.log_height()))]
    pub(crate) fn compute_round_poly(&mut self, round: usize) -> CompressedRoundPoly<Challenge> {
        if round < self.starting_round {
            return CompressedRoundPoly::default();
        }

        let FieldArray([coeff_0, coeff_2]) = self
            .trace
            .par_row_chunks(2)
            .zip(self.weight.par_chunks(2))
            .map(|(main, weight)| {
                let lo = dot_product::<Challenge, _, _>(cloned(self.gamma_powers), main.row(0));
                let hi = dot_product::<Challenge, _, _>(cloned(self.gamma_powers), main.row(1));
                let weight_lo = weight[0];
                let weight_hi = weight[1];
                FieldArray([lo * weight_lo, (hi - lo) * (weight_hi - weight_lo)])
            })
            .sum();
        let coeff_1 = {
            let eval_1 = self.claim - coeff_0;
            eval_1 - coeff_0 - coeff_2
        };

        let round_poly = RoundPoly(vec![coeff_0, coeff_1, coeff_2]);
        self.round_poly = round_poly.clone();
        round_poly.into_compressed()
    }

    pub(crate) fn fix_var(&mut self, round: usize, z_i: Challenge) {
        if round < self.starting_round {
            return;
        }

        self.trace = fix_var(self.trace.as_view(), z_i);
        self.weight = fix_var(RowMajorMatrixView::new_col(&self.weight), z_i).values;
        self.claim = self.round_poly.subclaim(z_i);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        self.trace.values
    }

    fn log_height(&self) -> usize {
        log2_strict_usize(self.trace.height())
    }
}
