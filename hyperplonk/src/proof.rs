use alloc::vec::Vec;

use itertools::chain;
use p3_field::Field;

use crate::horner;

#[derive(Debug)]
pub struct Proof<Challenge> {
    // pub fractional_sumcheck: FractionalSumcheckProof<Challenge>,
    pub zero_sumcheck: SumcheckProof<Challenge>,
}

#[derive(Debug)]
pub struct SumcheckProof<Challenge> {
    pub num_vars: usize,
    pub compressed_round_polys: Vec<CompressedRoundPoly<Challenge>>,
    pub evals: Vec<Challenge>,
}

#[derive(Debug)]
pub struct CompressedRoundPoly<Challenge>(pub Vec<Challenge>);

impl<Challenge: Field> CompressedRoundPoly<Challenge> {
    pub fn subclaim(
        &self,
        claim: Challenge,
        z_i: Challenge,
        r_i: Challenge,
        r_i_inv: Challenge,
    ) -> Challenge {
        let eval_1 = (claim - (Challenge::ONE - r_i) * self.0[0]) * r_i_inv;
        let coeff_1 = eval_1 - self.0.iter().copied().sum::<Challenge>();
        horner(chain![[&self.0[0], &coeff_1], &self.0[1..]], z_i)
    }
}
