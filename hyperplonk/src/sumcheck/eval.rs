use alloc::vec;
use alloc::vec::Vec;

use itertools::cloned;
use p3_field::{Field, dot_product};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{CompressedRoundPoly, RingArray, RoundPoly, fix_var};

pub(crate) struct EvalProver<'a, Challenge> {
    pub(crate) trace: RowMajorMatrix<Challenge>,
    pub(crate) weight: Vec<Challenge>,
    pub(crate) gamma_powers: &'a [Challenge],
    pub(crate) claim: Challenge,
    pub(crate) round_poly: RoundPoly<Challenge>,
}

impl<Challenge: Field> EvalProver<'_, Challenge> {
    #[instrument(
        level = "debug",
        name = "compute eval round poly",
        skip_all,
        fields(log_b = log_b)
    )]
    pub(crate) fn compute_round_poly(&mut self, log_b: usize) -> CompressedRoundPoly<Challenge> {
        if log_b + 1 != log2_strict_usize(self.trace.height()) {
            return CompressedRoundPoly(vec![Challenge::ZERO; 2]);
        }

        let RingArray([coeff_0, coeff_2]) = self
            .trace
            .par_row_chunks(2)
            .zip(self.weight.par_chunks(2))
            .map(|(main, weight)| {
                let lo: Challenge =
                    dot_product(cloned(self.gamma_powers), main.row(0).unwrap().into_iter());
                let hi: Challenge =
                    dot_product(cloned(self.gamma_powers), main.row(1).unwrap().into_iter());
                let weight_lo = weight[0];
                let weight_hi = weight[1];
                RingArray([lo * weight_lo, (hi - lo) * (weight_hi - weight_lo)])
            })
            .sum();
        let compressed_round_poly = CompressedRoundPoly(vec![coeff_0, coeff_2]);

        self.round_poly = RoundPoly::from_compressed(self.claim, compressed_round_poly.clone());

        compressed_round_poly
    }

    pub(crate) fn fix_var(&mut self, log_b: usize, z_i: Challenge) {
        if log_b + 1 != log2_strict_usize(self.trace.height()) {
            return;
        }

        self.trace = fix_var(self.trace.as_view(), z_i);
        self.weight = fix_var(RowMajorMatrixView::new_col(&self.weight), z_i).values;
        self.claim = self.round_poly.subclaim(z_i);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        self.trace.values
    }
}
