use alloc::vec::Vec;

use itertools::chain;
use p3_field::Field;

use crate::horner;

#[derive(Clone, Debug)]
pub struct Proof<Challenge> {
    pub log_heights: Vec<usize>,
    // pub fractional_sum_check: FractionalSumCheckProof<Challenge>,
    pub zero_check: ZeroCheckProof<Challenge>,
}

#[derive(Clone, Debug)]
pub struct ZeroCheckProof<Challenge> {
    pub univariate_skips: Vec<UnivariateSkipProof<Challenge>>,
    pub regular_sumcheck: BatchSumcheckProof<Challenge>,
    pub univariate_eval_sumcheck: BatchSumcheckProof<Challenge>,
}

#[derive(Clone, Debug, Default)]
pub struct UnivariateSkipProof<Challenge> {
    pub skip_rounds: usize,
    pub round_poly: RoundPoly<Challenge>,
}

#[derive(Clone, Debug, Default)]
pub struct BatchSumcheckProof<Challenge> {
    pub compressed_round_polys: Vec<CompressedRoundPoly<Challenge>>,
    pub evals: Vec<Vec<Challenge>>,
}

#[derive(Clone, Debug, Default)]

pub struct RoundPoly<Challenge>(pub Vec<Challenge>);

impl<Challenge: Field> RoundPoly<Challenge> {
    pub fn subclaim(&self, z_i: Challenge) -> Challenge {
        horner(&self.0, z_i)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompressedRoundPoly<Challenge>(pub Vec<Challenge>);

impl<Challenge: Field> CompressedRoundPoly<Challenge> {
    pub fn eq_weighted_subclaim(
        &self,
        claim: Challenge,
        z_i: Challenge,
        r_i: Challenge,
        r_i_inv: Challenge,
    ) -> Challenge {
        let (coeff_0, coeffs_rest) = self
            .0
            .split_first()
            .map(|(coeff_0, coeffs_rest)| (*coeff_0, coeffs_rest))
            .unwrap_or((Challenge::ZERO, [].as_slice()));
        let eval_1 = (claim - (Challenge::ONE - r_i) * coeff_0) * r_i_inv;
        let coeff_1 = eval_1 - self.0.iter().copied().sum::<Challenge>();
        horner(chain![[&coeff_0, &coeff_1], coeffs_rest], z_i)
    }

    pub fn subclaim(&self, claim: Challenge, z_i: Challenge) -> Challenge {
        let (coeff_0, coeffs_rest) = self
            .0
            .split_first()
            .map(|(coeff_0, coeffs_rest)| (*coeff_0, coeffs_rest))
            .unwrap_or((Challenge::ZERO, [].as_slice()));
        let eval_1 = claim - coeff_0;
        let coeff_1 = eval_1 - self.0.iter().copied().sum::<Challenge>();
        horner(chain![[&coeff_0, &coeff_1], coeffs_rest], z_i)
    }
}
