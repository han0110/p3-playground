use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, chain, cloned, izip};
use p3_field::{Field, dot_product};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{CompressedRoundPoly, FieldSlice, RoundPoly, fix_var, vec_add};

pub(crate) struct EvalSumcheckProver<'a, Challenge> {
    pub(crate) claim: Challenge,
    pub(crate) trace: RowMajorMatrix<Challenge>,
    pub(crate) weight: Vec<Challenge>,
    pub(crate) gamma_powers: &'a [Challenge],
    pub(crate) starting_round: usize,
    pub(crate) round_poly: RoundPoly<Challenge>,
}

impl<Challenge: Field> EvalSumcheckProver<'_, Challenge> {
    #[instrument(skip_all, name = "compute eval round poly", fields(dim = %self.dim()))]
    pub(crate) fn compute_round_poly(&mut self, round: usize) -> CompressedRoundPoly<Challenge> {
        if round < self.starting_round {
            return CompressedRoundPoly::default();
        }

        let mut extra_evals = self
            .trace
            .par_row_chunks(2)
            .zip(self.weight.par_chunks(2))
            .par_fold_reduce(
                || vec![Challenge::ZERO; 2],
                |mut sum, (main, weight)| {
                    let lo = main.row(0);
                    let hi = main.row(1);
                    let mut eval = hi.collect_vec();
                    let diff = izip!(&eval, lo).map(|(hi, lo)| *hi - lo).collect_vec();
                    let mut weight_eval = weight[1];
                    let weight_diff = weight[1] - weight[0];
                    sum.iter_mut().enumerate().for_each(|(d, sum)| {
                        if d > 0 {
                            eval.slice_add_assign(&diff);
                            weight_eval += weight_diff;
                        }
                        *sum += dot_product::<Challenge, _, _>(
                            cloned(self.gamma_powers),
                            cloned(&eval),
                        ) * weight_eval;
                    });
                    sum
                },
                vec_add,
            )
            .into_iter();
        let eval_1 = extra_evals.next().unwrap();
        let eval_0 = self.claim - eval_1;
        let round_poly = RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals]);
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

    fn dim(&self) -> usize {
        log2_strict_usize(self.trace.height())
    }
}
