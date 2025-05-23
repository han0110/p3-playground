use alloc::vec::Vec;
use core::mem::swap;
use core::ops::{Add, Mul};

use itertools::chain;
use p3_field::{Field, batch_multiplicative_inverse, dot_product};
use p3_ml_pcs::MlPcs;
use serde::{Deserialize, Serialize};

use crate::{FieldSlice, HyperPlonkGenericConfig, evaluate_uv_poly};

type Com<C> = <<C as HyperPlonkGenericConfig>::Pcs as MlPcs<
    <C as HyperPlonkGenericConfig>::Challenge,
    <C as HyperPlonkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<C> = <<C as HyperPlonkGenericConfig>::Pcs as MlPcs<
    <C as HyperPlonkGenericConfig>::Challenge,
    <C as HyperPlonkGenericConfig>::Challenger,
>>::Proof;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<C: HyperPlonkGenericConfig> {
    pub log_bs: Vec<usize>,
    pub commitment: Com<C>,
    pub piop: PiopProof<C::Challenge>,
    pub pcs: PcsProof<C>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiopProof<Challenge> {
    pub fractional_sum: FractionalSumProof<Challenge>,
    pub air: AirProof<Challenge>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FractionalSumProof<Challenge> {
    pub sums: Vec<Vec<Fraction<Challenge>>>,
    pub layers: Vec<BatchSumcheckProof<Challenge>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fraction<Challenge> {
    pub numer: Challenge,
    pub denom: Challenge,
}

impl<Challenge: Field> Fraction<Challenge> {
    pub fn sum<'a>(fracs: impl IntoIterator<Item = &'a Self>) -> Option<Challenge> {
        let (numers, denoms) = fracs
            .into_iter()
            .map(|fraction| (fraction.numer, fraction.denom))
            .collect::<(Vec<_>, Vec<_>)>();
        if denoms.contains(&Challenge::ZERO) {
            return None;
        }
        Some(dot_product(
            numers.into_iter(),
            batch_multiplicative_inverse(&denoms).into_iter(),
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AirProof<Challenge> {
    pub univariate_skips: Vec<AirUnivariateSkipProof<Challenge>>,
    pub regular: BatchSumcheckProof<Challenge>,
    pub univariate_eval_check: BatchSumcheckProof<Challenge>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AirUnivariateSkipProof<Challenge> {
    pub skip_rounds: usize,
    pub zero_check_round_poly: RoundPoly<Challenge>,
    pub eval_check_round_poly: RoundPoly<Challenge>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BatchSumcheckProof<Challenge> {
    pub compressed_round_polys: Vec<CompressedRoundPoly<Challenge>>,
    pub evals: Vec<Vec<Challenge>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoundPoly<Challenge>(pub Vec<Challenge>);

impl<Challenge: Field> RoundPoly<Challenge> {
    pub fn subclaim(&self, z_i: Challenge) -> Challenge {
        evaluate_uv_poly(&self.0, z_i)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompressedRoundPoly<Challenge>(pub Vec<Challenge>);

impl<Challenge: Field> CompressedRoundPoly<Challenge> {
    pub fn random_linear_combine(compressed_round_polys: Vec<Self>, r: Challenge) -> Self {
        let mut compressed_round_polys = compressed_round_polys.into_iter();
        let init = compressed_round_polys.next().unwrap_or_default();
        compressed_round_polys
            .zip(r.powers().skip(1))
            .fold(init, |acc, (item, scalar)| acc + item * scalar)
    }

    pub fn subclaim(&self, claim: Challenge, z_i: Challenge) -> Challenge {
        let (coeff_0, coeffs_rest) = self
            .0
            .split_first()
            .map(|(coeff_0, coeffs_rest)| (*coeff_0, coeffs_rest))
            .unwrap_or((Challenge::ZERO, [].as_slice()));
        let eval_1 = claim - coeff_0;
        let coeff_1 = eval_1 - self.0.iter().copied().sum::<Challenge>();
        evaluate_uv_poly(chain![[&coeff_0, &coeff_1], coeffs_rest], z_i)
    }
}

impl<Challenge: Field> Add for CompressedRoundPoly<Challenge> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        if self.0.len() < rhs.0.len() {
            swap(&mut self, &mut rhs);
        }
        self.0[..rhs.0.len()].slice_add_assign(&rhs.0);
        self
    }
}

impl<Challenge: Field> Mul<Challenge> for CompressedRoundPoly<Challenge> {
    type Output = Self;

    fn mul(mut self, rhs: Challenge) -> Self::Output {
        self.0.slice_scale(rhs);
        self
    }
}
