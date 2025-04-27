use alloc::vec::Vec;
use core::cmp::min;
use core::marker::PhantomData;

use itertools::{Itertools, chain, cloned};
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    batch_multiplicative_inverse, dot_product,
};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{CompressedRoundPoly, RoundPoly, eq_poly, vander_mat_inv};

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

    pub(crate) fn into_compressed(mut self) -> CompressedRoundPoly<Challenge> {
        if self.0.len() > 1 {
            self.0.remove(1);
        }
        CompressedRoundPoly(self.0)
    }
}

pub(crate) struct EqHelper<'a, Val: Field, Challenge: ExtensionField<Val>> {
    evals: Vec<Challenge::ExtensionPacking>,
    r: &'a [Challenge],
    one_minus_r_inv: Vec<Challenge>,
    correcting_factors: Vec<Challenge>,
    _marker: PhantomData<Val>,
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>> EqHelper<'a, Val, Challenge> {
    pub(crate) fn new(r: &'a [Challenge]) -> Self {
        let evals = {
            let r = &r[min(r.len(), 1)..];
            let log_packing_width = log2_strict_usize(Val::Packing::WIDTH);
            let (r_lo, r_hi) = r.split_at(r.len().saturating_sub(log_packing_width));
            let mut evals_hi = eq_poly(r_hi, Challenge::ONE);
            evals_hi.resize(Val::Packing::WIDTH, Challenge::ZERO);
            eq_poly(r_lo, <_>::from_ext_slice(&evals_hi))
        };
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

    pub(crate) fn evals_packed(
        &self,
        round: usize,
    ) -> impl IndexedParallelIterator<Item = Challenge::ExtensionPacking> {
        self.evals.par_iter().step_by(1 << round).copied()
    }

    fn evals(&self, round: usize) -> impl IndexedParallelIterator<Item = Challenge> {
        let len = min(Val::Packing::WIDTH, 1 << (self.r.len() - 1));
        let step = len >> (self.r.len() - 1 - round);
        let eval = self.evals[0];
        (0..len).into_par_iter().step_by(step).map(move |i| {
            Challenge::from_basis_coefficients_fn(|j| {
                eval.as_basis_coefficients_slice()[j].as_slice()[i]
            })
        })
    }

    fn eval_0(&self, round: usize, claim: Challenge, eval_1: Challenge) -> Challenge {
        (claim - self.r[round] * eval_1) * self.one_minus_r_inv[round]
    }

    fn correcting_factor(&self, round: usize) -> Challenge {
        self.correcting_factors[round]
    }
}
