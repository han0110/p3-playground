use alloc::vec;
use alloc::vec::Vec;
use core::cmp::max;

use itertools::{Itertools, cloned};
use p3_air::Air;
use p3_commit::{LagrangeSelectors, PolynomialSpace};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, PackedValue, PrimeCharacteristicRing, TwoAdicField,
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_ml_pcs::eq_poly_packed;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::{
    AirMeta, AirRegularProver, EqHelper, EvalProver, FieldSlice, PackedExtensionValue,
    ProverConstraintFolderGeneric, ProverConstraintFolderOnExtension,
    ProverConstraintFolderOnExtensionPacking, ProverConstraintFolderOnPacking, RoundPoly, Trace,
    vec_pair_add,
};

pub(crate) struct AirUnivariateSkipProver<'a, Val: Field, Challenge: ExtensionField<Val>, A> {
    meta: &'a AirMeta,
    air: &'a A,
    public_values: &'a [Val],
    trace: Trace<Val, Challenge>,
    beta_powers: &'a [Challenge],
    gamma_powers: &'a [Challenge],
    alpha_powers: &'a [Challenge],
    skip_rounds: usize,
    zero_check_round_poly: RoundPoly<Challenge>,
    eval_check_round_poly: RoundPoly<Challenge>,
}

impl<'a, Val, Challenge, A> AirUnivariateSkipProver<'a, Val, Challenge, A>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        meta: &'a AirMeta,
        air: &'a A,
        public_values: &'a [Val],
        trace: Trace<Val, Challenge>,
        beta_powers: &'a [Challenge],
        gamma_powers: &'a [Challenge],
        alpha_powers: &'a [Challenge],
        skip_rounds: usize,
    ) -> Self {
        Self {
            meta,
            air,
            public_values,
            trace,
            beta_powers,
            gamma_powers,
            alpha_powers,
            skip_rounds,
            zero_check_round_poly: Default::default(),
            eval_check_round_poly: Default::default(),
        }
    }

    #[instrument(name = "compute univariate skip round poly", skip_all, fields(log_b = %self.trace.log_b()))]
    pub(crate) fn compute_round_poly(
        &mut self,
        z_zc: &[Challenge],
        z_fs: &[Challenge],
    ) -> (RoundPoly<Challenge>, RoundPoly<Challenge>) {
        let Trace::Packing(trace) = &self.trace else {
            unimplemented!()
        };

        let (zero_check_poly, eval_check_poly) = {
            let has_interaction = self.meta.has_interaction();
            let zero_check_quotient_degree = self.meta.zero_check_uv_degree.saturating_sub(1);
            let eval_check_degree = self.meta.eval_check_uv_degree;
            let added_bits = log2_ceil_usize(max(zero_check_quotient_degree, eval_check_degree));
            let sels = domain(self.skip_rounds)
                .selectors_on_coset(quotient_domain(self.skip_rounds, added_bits));
            let zero_check_eq_packed = eq_poly_packed(z_zc);
            let eval_check_eq_packed = eq_poly_packed(z_fs);
            let (zero_check_values, eval_check_values) = trace
                .par_row_chunks(1 << self.skip_rounds)
                .enumerate()
                .par_fold_reduce(
                    || {
                        let len = 1 << (self.skip_rounds + added_bits);
                        (
                            vec![<_>::ZERO; len],
                            vec![<_>::ZERO; has_interaction.then_some(len).unwrap_or_default()],
                        )
                    },
                    |(mut zero_check_values, mut eval_check_values), (chunk, trace)| {
                        self.compute_chunk_values(
                            &mut zero_check_values,
                            &mut eval_check_values,
                            trace,
                            zero_check_eq_packed[chunk],
                            has_interaction.then(|| eval_check_eq_packed[chunk]),
                            &sels,
                            added_bits,
                            chunk == 0,
                            chunk == zero_check_eq_packed.len() - 1,
                        );
                        (zero_check_values, eval_check_values)
                    },
                    vec_pair_add,
                );

            let values_to_poly = |values: Vec<_>, degree| {
                let mut poly = Radix2DitParallel::default()
                    .coset_idft_algebra_batch(
                        RowMajorMatrix::new_col(values.par_iter().map(<_>::ext_sum).collect()),
                        Val::GENERATOR,
                    )
                    .values;
                poly.truncate(degree << self.skip_rounds);
                poly
            };
            join(
                || values_to_poly(zero_check_values, zero_check_quotient_degree),
                || {
                    self.meta
                        .has_interaction()
                        .then(|| values_to_poly(eval_check_values, eval_check_degree))
                        .unwrap_or_default()
                },
            )
        };

        let zero_check_round_poly = RoundPoly(zero_check_poly);
        let eval_check_round_poly = RoundPoly(eval_check_poly);
        self.zero_check_round_poly = zero_check_round_poly.clone();
        self.eval_check_round_poly = eval_check_round_poly.clone();
        (zero_check_round_poly, eval_check_round_poly)
    }

    pub(crate) fn to_regular_prover(
        &self,
        x: Challenge,
        zero_check_eq_helper: &'a EqHelper<'a, Val, Challenge>,
        z_fs: &'a [Challenge],
    ) -> AirRegularProver<'a, Val, Challenge, A>
    where
        A: for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>
            + for<'t> Air<ProverConstraintFolderOnExtension<'t, Val, Challenge>>
            + for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val, Challenge>>,
    {
        let zero_check_claim = self.zero_check_round_poly.subclaim(x)
            * (x.exp_power_of_2(self.skip_rounds) - Val::ONE);
        let eval_check_claim = self.eval_check_round_poly.subclaim(x);
        let trace = {
            let lagrange_evals = lagrange_evals(self.skip_rounds, x);
            self.trace.fix_lo_scalars(&lagrange_evals)
        };
        let sels = selectors_at_point(self.skip_rounds, x);
        let mut regular_prover = AirRegularProver::new(
            self.meta,
            self.air,
            self.public_values,
            trace,
            self.beta_powers,
            self.gamma_powers,
            self.alpha_powers,
            zero_check_claim,
            eval_check_claim,
            zero_check_eq_helper,
            z_fs,
        );
        regular_prover.is_first_row.0 = sels.is_first_row;
        regular_prover.is_last_row.0 = sels.is_last_row;
        regular_prover.is_transition.0 = sels.is_transition;
        regular_prover
    }

    pub(crate) fn into_univariate_eval_prover<'b>(
        self,
        x: Challenge,
        z: &[Challenge],
        evals: &[Challenge],
        gamma_powers: &'b [Challenge],
    ) -> EvalProver<'b, Challenge> {
        let claim = dot_product(cloned(evals), cloned(gamma_powers));
        let trace = match self.trace.fix_hi_vars(z) {
            Trace::Extension(trace) => trace,
            _ => unimplemented!(),
        };
        let weight = lagrange_evals(self.skip_rounds, x);
        EvalProver {
            trace,
            weight,
            gamma_powers,
            claim,
            round_poly: Default::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_chunk_values(
        &self,
        zero_check_values: &mut [Challenge::ExtensionPacking],
        eval_check_values: &mut [Challenge::ExtensionPacking],
        trace: RowMajorMatrixView<Val::Packing>,
        zero_check_eq_eval: Challenge::ExtensionPacking,
        eval_check_eq_eval: Option<Challenge::ExtensionPacking>,
        sels: &LagrangeSelectors<Vec<Val>>,
        added_bits: usize,
        is_first_chunk: bool,
        is_last_chunk: bool,
    ) where
        Val: TwoAdicField + Ord,
        Challenge: ExtensionField<Val>,
        A: for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>,
    {
        let trace_lde = {
            let values = Val::Packing::unpack_slice(trace.values);
            let mut lde = RowMajorMatrix::new(
                Vec::with_capacity(values.len() << added_bits),
                trace.width() * Val::Packing::WIDTH,
            );
            lde.values.extend(cloned(values));
            Radix2DitParallel::default().coset_lde_batch(lde, added_bits, Val::GENERATOR)
        };
        if self.meta.has_interaction() {
            let eval_check_eq_eval = eval_check_eq_eval.unwrap();
            zero_check_values
                .par_iter_mut()
                .zip(eval_check_values)
                .enumerate()
                .for_each(|(row, (zero_check, eval_check))| {
                    let (zero_check_accumulator, eval_check_accumulator) =
                        self.eval(&trace_lde, sels, row, is_first_chunk, is_last_chunk);
                    *zero_check += zero_check_accumulator * zero_check_eq_eval;
                    *eval_check += eval_check_accumulator * eval_check_eq_eval;
                });
        } else {
            zero_check_values
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, zero_check)| {
                    let (zero_check_accumulator, _) =
                        self.eval(&trace_lde, sels, row, is_first_chunk, is_last_chunk);
                    *zero_check += zero_check_accumulator * zero_check_eq_eval;
                });
        }
    }

    #[inline]
    fn eval(
        &self,
        trace_lde: &impl Matrix<Val>,
        sels: &LagrangeSelectors<Vec<Val>>,
        row: usize,
        is_first_chunk: bool,
        is_last_chunk: bool,
    ) -> (Challenge::ExtensionPacking, Challenge::ExtensionPacking) {
        let row_slice = trace_lde.row_slice(row).unwrap();
        let main = RowMajorMatrixView::new(Val::Packing::pack_slice(&row_slice), self.meta.width);
        let sels = selectors_at_row(sels, is_first_chunk, is_last_chunk, row);
        let (zero_check_alpha_powers, eval_check_alpha_powers) =
            self.alpha_powers.split_at(self.meta.constraint_count);
        let mut folder = ProverConstraintFolderGeneric {
            main,
            public_values: self.public_values,
            is_first_row: sels.is_first_row,
            is_last_row: sels.is_last_row,
            is_transition: sels.is_transition,
            beta_powers: self.beta_powers,
            gamma_powers: self.gamma_powers,
            zero_check_alpha_powers,
            eval_check_alpha_powers,
            zero_check_accumulator: Challenge::ExtensionPacking::ZERO,
            eval_check_accumulator: Challenge::ExtensionPacking::ZERO,
            constraint_index: 0,
            interaction_index: 0,
        };
        self.air.eval(&mut folder);
        (
            folder.zero_check_accumulator * sels.inv_vanishing,
            folder.eval_check_accumulator,
        )
    }
}

#[inline]
fn domain<Val: TwoAdicField>(skip_rounds: usize) -> TwoAdicMultiplicativeCoset<Val> {
    TwoAdicMultiplicativeCoset::new(Val::ONE, skip_rounds).unwrap()
}

#[inline]
fn quotient_domain<Val: TwoAdicField>(
    skip_rounds: usize,
    added_bits: usize,
) -> TwoAdicMultiplicativeCoset<Val> {
    TwoAdicMultiplicativeCoset::new(Val::GENERATOR, skip_rounds + added_bits).unwrap()
}

#[inline]
fn selectors_at_row<Val: Field>(
    sels: &LagrangeSelectors<Vec<Val>>,
    is_first_chunk: bool,
    is_last_chunk: bool,
    row: usize,
) -> LagrangeSelectors<Val::Packing> {
    let mut v = LagrangeSelectors {
        is_first_row: Val::Packing::ZERO,
        is_last_row: Val::Packing::ZERO,
        is_transition: Val::Packing::ONE,
        inv_vanishing: Val::Packing::from(sels.inv_vanishing[row]),
    };
    if is_first_chunk {
        v.is_first_row.as_slice_mut()[0] = sels.is_first_row[row]
    }
    if is_last_chunk {
        v.is_last_row.as_slice_mut()[Val::Packing::WIDTH - 1] = sels.is_last_row[row];
        v.is_transition.as_slice_mut()[Val::Packing::WIDTH - 1] = sels.is_transition[row];
    }
    v
}

#[inline]
pub(crate) fn selectors_at_point<Val: TwoAdicField, Challenge: ExtensionField<Val>>(
    skip_rounds: usize,
    x: Challenge,
) -> LagrangeSelectors<Challenge> {
    let mut sels = domain(skip_rounds).selectors_at_point(x);
    if skip_rounds == 0 {
        sels.is_transition = Challenge::ZERO;
    }
    sels
}

pub(crate) fn lagrange_evals<Val: TwoAdicField, Challenge: ExtensionField<Val>>(
    skip_rounds: usize,
    z: Challenge,
) -> Vec<Challenge> {
    let subgroup = cyclic_subgroup_coset_known_order(
        Val::two_adic_generator(skip_rounds),
        Val::ONE,
        1 << skip_rounds,
    )
    .collect_vec();
    let vanishing_over_height = (z.exp_power_of_2(skip_rounds) - Challenge::ONE)
        * Val::ONE.halve().exp_u64(skip_rounds as u64);
    let diff_invs =
        batch_multiplicative_inverse(&subgroup.par_iter().map(|&x| z - x).collect::<Vec<_>>());
    subgroup
        .into_par_iter()
        .zip(diff_invs)
        .map(|(x, diff_inv)| vanishing_over_height * diff_inv * x)
        .collect()
}

pub(crate) fn evaluations_on_domain<Val: TwoAdicField + Ord, Challenge: ExtensionField<Val>>(
    skip_rounds: usize,
    poly: &RoundPoly<Challenge>,
) -> Vec<Challenge> {
    let mut coeffs = vec![Challenge::ZERO; 1 << skip_rounds];
    poly.0
        .chunks(1 << skip_rounds)
        .for_each(|chunk| coeffs[..chunk.len()].slice_add_assign(chunk));
    Radix2DitParallel::default().dft_algebra(coeffs)
}
