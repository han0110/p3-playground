use alloc::vec;
use alloc::vec::Vec;

use itertools::{chain, cloned};
use p3_air::{Air, BaseAir};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::DenseMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::{
    AirMeta, AirTrace, CompressedRoundPoly, EqHelper, ExtensionPacking, FieldSlice,
    PackedExtensionValue, ProverConstraintFolderGeneric, ProverConstraintFolderOnExtension,
    ProverConstraintFolderOnExtensionPacking, ProverConstraintFolderOnPacking, RoundPoly, eq_eval,
};

pub(crate) struct RegularSumcheckProver<'a, Val: Field, Challenge: ExtensionField<Val>, A> {
    pub(crate) meta: &'a AirMeta,
    pub(crate) air: &'a A,
    pub(crate) public_values: &'a [Val],
    pub(crate) zero_check_claim: Challenge,
    pub(crate) eval_check_claim: Challenge,
    pub(crate) trace: AirTrace<Val, Challenge>,
    pub(crate) beta_powers: &'a [Challenge],
    pub(crate) gamma_powers: &'a [Challenge],
    pub(crate) alpha_powers: &'a [Challenge],
    pub(crate) eq_r_helper: &'a EqHelper<'a, Val, Challenge>,
    pub(crate) eq_z_fs_helper: &'a EqHelper<'a, Val, Challenge>,
    pub(crate) starting_round: usize,
    pub(crate) is_first_row: IsFirstRow<Challenge>,
    pub(crate) is_last_row: IsLastRow<Challenge>,
    pub(crate) is_transition: IsTransition<Challenge>,
    pub(crate) zero_check_round_poly: RoundPoly<Challenge>,
    pub(crate) eval_check_round_poly: RoundPoly<Challenge>,
    pub(crate) eq_r_z_eval: Challenge,
    pub(crate) eq_z_fs_z_eval: Challenge,
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>, A> RegularSumcheckProver<'a, Val, Challenge, A>
where
    A: BaseAir<Val>
        + for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        meta: &'a AirMeta,
        air: &'a A,
        public_values: &'a [Val],
        zero_check_claim: Challenge,
        eval_check_claim: Challenge,
        trace: AirTrace<Val, Challenge>,
        beta_powers: &'a [Challenge],
        gamma_powers: &'a [Challenge],
        alpha_powers: &'a [Challenge],
        eq_r_helper: &'a EqHelper<'a, Val, Challenge>,
        eq_z_fs_helper: &'a EqHelper<'a, Val, Challenge>,
        max_regular_rounds: usize,
    ) -> Self {
        let starting_round = max_regular_rounds - trace.log_height();
        Self {
            air,
            meta,
            public_values,
            zero_check_claim,
            eval_check_claim,
            trace,
            beta_powers,
            gamma_powers,
            alpha_powers,
            starting_round,
            eq_r_helper,
            eq_z_fs_helper,
            is_first_row: IsFirstRow(Challenge::ONE),
            is_last_row: IsLastRow(Challenge::ONE),
            is_transition: IsTransition(Challenge::ZERO),
            zero_check_round_poly: Default::default(),
            eval_check_round_poly: Default::default(),
            eq_r_z_eval: Challenge::ONE,
            eq_z_fs_z_eval: Challenge::ONE,
        }
    }

    pub(crate) fn compute_round_poly(&mut self, round: usize) -> CompressedRoundPoly<Challenge> {
        if round < self.starting_round {
            return CompressedRoundPoly::default();
        }

        let (zero_check_round_poly, eval_check_round_poly) = match &self.trace {
            AirTrace::Packing(_) => self.compute_eq_weighted_round_poly_packing(round),
            AirTrace::ExtensionPacking(_) => {
                self.compute_eq_weighted_round_poly_extension_packing(round)
            }
            AirTrace::Extension(_) => self.compute_eq_weighted_round_poly_extension(round),
        };

        self.zero_check_round_poly = zero_check_round_poly.clone();
        self.eval_check_round_poly = eval_check_round_poly.clone();

        zero_check_round_poly
            .mul_by_scaled_eq(self.eq_r_z_eval, self.eq_r_helper.r_i(round))
            .into_compressed()
            + eval_check_round_poly
                .mul_by_scaled_eq(self.eq_z_fs_z_eval, self.eq_z_fs_helper.r_i(round))
                .into_compressed()
    }

    #[instrument(skip_all, name = "compute eq weighted round poly (packing)", fields(log_h = %self.trace.log_height()))]
    fn compute_eq_weighted_round_poly_packing(
        &mut self,
        round: usize,
    ) -> (RoundPoly<Challenge>, RoundPoly<Challenge>) {
        let AirTrace::Packing(trace) = &self.trace else {
            unreachable!()
        };

        let has_interaction = self.meta.has_interaction();
        let degree = self.meta.regular_sumcheck_degree();
        let is_first_row = IsFirstRow(Val::ONE).eval_packed();
        let is_last_row = IsLastRow(Val::ONE).eval_packed();
        let is_transition = IsTransition(Val::ZERO).eval_packed();

        let (zero_check_extra_evals, eval_check_extra_evals) = (0..trace.height() / 2)
            .into_par_iter()
            .par_fold_reduce(
                || self.eval_state(),
                |mut state, row| {
                    state.init(
                        trace,
                        is_first_row,
                        is_last_row,
                        is_transition,
                        self.eq_r_helper.eval_packed(round, row),
                        has_interaction.then(|| self.eq_z_fs_helper.eval_packed(round, row)),
                        row,
                    );
                    state.eval_and_accumulate();
                    (1..degree).for_each(|_| {
                        state.next_point();
                        state.eval_and_accumulate();
                    });
                    state
                },
                EvalsAccumulator::sum,
            )
            .into_evals();

        self.recover_eq_weighted_round_poly(
            round,
            zero_check_extra_evals.iter().map(<_>::ext_sum),
            eval_check_extra_evals.iter().map(<_>::ext_sum),
        )
    }

    #[instrument(skip_all, name = "compute eq weighted round poly (ext packing)", fields(log_h = %self.trace.log_height()))]
    fn compute_eq_weighted_round_poly_extension_packing(
        &mut self,
        round: usize,
    ) -> (RoundPoly<Challenge>, RoundPoly<Challenge>) {
        let AirTrace::ExtensionPacking(trace) = &self.trace else {
            unreachable!()
        };

        let has_interaction = self.meta.has_interaction();
        let degree = self.meta.regular_sumcheck_degree();
        let is_first_row = self.is_first_row.eval_packed();
        let is_last_row = self.is_last_row.eval_packed();
        let is_transition = self.is_transition.eval_packed();

        let (zero_check_extra_evals, eval_check_extra_evals) = (0..trace.height() / 2)
            .into_par_iter()
            .par_fold_reduce(
                || self.eval_state(),
                |mut state, row| {
                    state.init(
                        trace,
                        is_first_row,
                        is_last_row,
                        is_transition,
                        self.eq_r_helper.eval_packed(round, row),
                        has_interaction.then(|| self.eq_z_fs_helper.eval_packed(round, row)),
                        row,
                    );
                    state.eval_packed_and_accumulate();
                    (1..degree).for_each(|_| {
                        state.next_point();
                        state.eval_packed_and_accumulate();
                    });
                    state
                },
                EvalsAccumulator::sum,
            )
            .into_evals();

        self.recover_eq_weighted_round_poly(
            round,
            zero_check_extra_evals.iter().map(<_>::ext_sum),
            eval_check_extra_evals.iter().map(<_>::ext_sum),
        )
    }

    #[instrument(skip_all, name = "compute eq weighted round poly (ext)", fields(log_h = %self.trace.log_height()))]
    fn compute_eq_weighted_round_poly_extension(
        &mut self,
        round: usize,
    ) -> (RoundPoly<Challenge>, RoundPoly<Challenge>) {
        let AirTrace::Extension(trace) = &self.trace else {
            unreachable!()
        };

        let has_interaction = self.meta.has_interaction();
        let degree = self.meta.regular_sumcheck_degree();

        let (zero_check_extra_evals, eval_check_extra_evals) = (0..trace.height() / 2)
            .into_par_iter()
            .par_fold_reduce(
                || self.eval_state(),
                |mut state, row| {
                    state.init(
                        trace,
                        self.is_first_row.0,
                        self.is_last_row.0,
                        self.is_transition.0,
                        self.eq_r_helper.eval(round, row),
                        has_interaction.then(|| self.eq_z_fs_helper.eval(round, row)),
                        row,
                    );
                    state.eval_and_accumulate();
                    (1..degree).for_each(|_| {
                        state.next_point();
                        state.eval_and_accumulate();
                    });
                    state
                },
                EvalsAccumulator::sum,
            )
            .into_evals();

        self.recover_eq_weighted_round_poly(round, zero_check_extra_evals, eval_check_extra_evals)
    }

    fn eval_state<
        Var: Copy + Send + Sync + PrimeCharacteristicRing,
        VarEF: Copy + Algebra<Var> + From<Challenge>,
    >(
        &self,
    ) -> EvalsAccumulator<Val, Challenge, Var, VarEF, A> {
        EvalsAccumulator::new(
            self.meta,
            self.air,
            self.public_values,
            self.beta_powers,
            self.gamma_powers,
            self.alpha_powers,
        )
    }

    fn recover_eq_weighted_round_poly(
        &self,
        round: usize,
        zero_check_extra_evals: impl IntoIterator<Item = Challenge>,
        eval_check_extra_evals: impl IntoIterator<Item = Challenge>,
    ) -> (RoundPoly<Challenge>, RoundPoly<Challenge>) {
        fn inner<Val: Field, Challenge: ExtensionField<Val>>(
            eq_helper: &EqHelper<Val, Challenge>,
            claim: Challenge,
            round: usize,
            extra_evals: impl IntoIterator<Item = Challenge>,
        ) -> RoundPoly<Challenge> {
            let mut extra_evals = extra_evals
                .into_iter()
                .map(|eval| eval * eq_helper.correcting_factor(round));
            let Some(eval_1) = extra_evals.next() else {
                return Default::default();
            };
            let eval_0 = eq_helper.eval_0(round, claim, eval_1);
            RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
        }

        (
            inner(
                self.eq_r_helper,
                self.zero_check_claim,
                round,
                zero_check_extra_evals,
            ),
            inner(
                self.eq_z_fs_helper,
                self.eval_check_claim,
                round,
                eval_check_extra_evals,
            ),
        )
    }

    pub(crate) fn fix_var(&mut self, round: usize, z_i: Challenge) {
        if round < self.starting_round {
            return;
        }

        self.zero_check_claim = self.zero_check_round_poly.subclaim(z_i);
        self.eval_check_claim = self.eval_check_round_poly.subclaim(z_i);
        self.trace = self.trace.fix_var(z_i);
        self.is_first_row = self.is_first_row.fix_var(z_i);
        self.is_last_row = self.is_last_row.fix_var(z_i);
        self.is_transition = self.is_transition.fix_var(z_i);
        self.eq_r_z_eval *= eq_eval([&self.eq_r_helper.r_i(round)], [&z_i]);
        self.eq_z_fs_z_eval *= eq_eval([&self.eq_z_fs_helper.r_i(round)], [&z_i]);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        let AirTrace::Extension(trace) = self.trace else {
            unreachable!()
        };
        trace.values
    }
}

struct EvalsAccumulator<'a, Val, Challenge, Var, VarEF, A> {
    meta: &'a AirMeta,
    air: &'a A,
    public_values: &'a [Val],
    beta_powers: &'a [Challenge],
    gamma_powers: &'a [Challenge],
    alpha_powers: &'a [Challenge],
    point_index: usize,
    main_eval: Vec<Var>,
    main_diff: Vec<Var>,
    is_first_row_diff: Option<Var>,
    is_first_row_eval: Var,
    is_last_row_diff: Option<Var>,
    is_last_row_eval: Var,
    is_transition_diff: Option<Var>,
    is_transition_eval: Var,
    eq_r_eval: VarEF,
    eq_z_fs_eval: Option<VarEF>,
    zero_check_evals: Vec<VarEF>,
    eval_check_evals: Vec<VarEF>,
}

impl<'a, Val, Challenge, Var, VarEF, A> EvalsAccumulator<'a, Val, Challenge, Var, VarEF, A>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Var: Copy + Send + Sync + PrimeCharacteristicRing,
    VarEF: Copy + Algebra<Var> + From<Challenge>,
{
    #[inline]
    fn new(
        meta: &'a AirMeta,
        air: &'a A,
        public_values: &'a [Val],
        beta_powers: &'a [Challenge],
        gamma_powers: &'a [Challenge],
        alpha_powers: &'a [Challenge],
    ) -> Self {
        Self {
            meta,
            air,
            public_values,
            beta_powers,
            gamma_powers,
            alpha_powers,
            point_index: 0,
            main_eval: vec![Var::ZERO; 2 * meta.width],
            main_diff: vec![Var::ZERO; 2 * meta.width],
            is_first_row_diff: None,
            is_first_row_eval: Var::ZERO,
            is_last_row_diff: None,
            is_last_row_eval: Var::ZERO,
            is_transition_diff: None,
            is_transition_eval: Var::ZERO,
            eq_r_eval: VarEF::ZERO,
            eq_z_fs_eval: None,
            zero_check_evals: vec![VarEF::ZERO; meta.regular_sumcheck_degree()],
            eval_check_evals: vec![
                VarEF::ZERO;
                meta.has_interaction()
                    .then(|| meta.regular_sumcheck_degree())
                    .unwrap_or_default()
            ],
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn init(
        &mut self,
        trace: &impl Matrix<Var>,
        is_first_row: Var,
        is_last_row: Var,
        is_transition: Var,
        eq_r_eval: VarEF,
        eq_z_fs_eval: Option<VarEF>,
        row: usize,
    ) {
        let last_row = trace.height() / 2 - 1;
        let hi = trace.row(2 * row + 1).unwrap();
        let lo = trace.row(2 * row).unwrap();
        self.point_index = 0;
        self.main_eval.slice_assign_iter(hi);
        self.main_diff.slice_sub_iter(cloned(&self.main_eval), lo);
        self.is_first_row_eval = Var::ZERO;
        self.is_first_row_diff = (row == 0).then(|| -is_first_row);
        self.is_last_row_diff = (row == last_row).then_some(is_last_row);
        self.is_last_row_eval = self.is_last_row_diff.unwrap_or_default();
        self.is_transition_eval = if row == last_row {
            is_transition
        } else {
            Var::ONE
        };
        self.is_transition_diff = (row == last_row).then(|| self.is_transition_eval - Var::ONE);
        self.eq_r_eval = eq_r_eval;
        self.eq_z_fs_eval = eq_z_fs_eval;
    }

    #[inline]
    fn next_point(&mut self) {
        self.point_index += 1;
        self.main_eval.slice_add_assign(&self.main_diff);
        if let Some(is_first_row_diff) = self.is_first_row_diff {
            self.is_first_row_eval += is_first_row_diff;
        }
        if let Some(is_last_row_diff) = self.is_last_row_diff {
            self.is_last_row_eval += is_last_row_diff;
        }
        if let Some(is_transition_diff) = self.is_transition_diff {
            self.is_transition_eval += is_transition_diff;
        }
    }

    #[inline]
    fn eval_and_accumulate(&mut self)
    where
        A: for<'t> Air<ProverConstraintFolderGeneric<'t, Val, Challenge, Var, VarEF>>,
        Var: Algebra<Val>,
    {
        let (zero_check_alpha_powers, eval_check_alpha_powers) =
            self.alpha_powers.split_at(self.meta.constraint_count);
        let mut builder = ProverConstraintFolderGeneric {
            main: DenseMatrix::new(&self.main_eval, self.meta.width),
            public_values: self.public_values,
            is_first_row: self.is_first_row_eval,
            is_last_row: self.is_last_row_eval,
            is_transition: self.is_transition_eval,
            beta_powers: self.beta_powers,
            gamma_powers: self.gamma_powers,
            zero_check_alpha_powers,
            eval_check_alpha_powers,
            zero_check_accumulator: VarEF::ZERO,
            eval_check_accumulator: VarEF::ZERO,
            constraint_index: 0,
            interaction_index: 0,
        };
        self.air.eval(&mut builder);
        self.zero_check_evals[self.point_index] += self.eq_r_eval * builder.zero_check_accumulator;
        if let Some(eq_z_fs_eval) = self.eq_z_fs_eval {
            self.eval_check_evals[self.point_index] +=
                eq_z_fs_eval * builder.eval_check_accumulator;
        }
    }

    fn sum(mut lhs: Self, rhs: Self) -> Self {
        lhs.zero_check_evals.slice_add_assign(&rhs.zero_check_evals);
        lhs.eval_check_evals.slice_add_assign(&rhs.eval_check_evals);
        lhs
    }

    fn into_evals(self) -> (Vec<VarEF>, Vec<VarEF>) {
        (self.zero_check_evals, self.eval_check_evals)
    }
}

impl<Val: Field, Challenge: ExtensionField<Val>, A>
    EvalsAccumulator<
        '_,
        Val,
        Challenge,
        Challenge::ExtensionPacking,
        Challenge::ExtensionPacking,
        A,
    >
{
    #[inline]
    fn eval_packed_and_accumulate(&mut self)
    where
        A: for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val, Challenge>>,
    {
        let (zero_check_alpha_powers, eval_check_alpha_powers) =
            self.alpha_powers.split_at(self.meta.constraint_count);
        let mut builder = ProverConstraintFolderOnExtensionPacking {
            main: DenseMatrix::new(
                ExtensionPacking::from_slice(&self.main_eval),
                self.meta.width,
            ),
            public_values: self.public_values,
            is_first_row: ExtensionPacking(self.is_first_row_eval),
            is_last_row: ExtensionPacking(self.is_last_row_eval),
            is_transition: ExtensionPacking(self.is_transition_eval),
            beta_powers: self.beta_powers,
            gamma_powers: self.gamma_powers,
            zero_check_alpha_powers,
            eval_check_alpha_powers,
            zero_check_accumulator: ExtensionPacking::ZERO,
            eval_check_accumulator: ExtensionPacking::ZERO,
            constraint_index: 0,
            interaction_index: 0,
        };
        self.air.eval(&mut builder);

        self.zero_check_evals[self.point_index] +=
            self.eq_r_eval * builder.zero_check_accumulator.0;
        if let Some(eq_z_fs_eval) = self.eq_z_fs_eval {
            self.eval_check_evals[self.point_index] +=
                eq_z_fs_eval * builder.eval_check_accumulator.0;
        }
    }
}

pub(crate) struct IsFirstRow<Challenge>(pub(crate) Challenge);

impl<Challenge: Field> IsFirstRow<Challenge> {
    #[must_use]
    fn fix_var(&self, z_i: Challenge) -> Self {
        Self(self.0 * (Challenge::ONE - z_i))
    }

    #[inline]
    fn eval_packed<Val: Field>(&self) -> Challenge::ExtensionPacking
    where
        Challenge: ExtensionField<Val>,
    {
        Challenge::ExtensionPacking::from_basis_coefficients_fn(|i| {
            Val::Packing::from_fn(|j| {
                (j == 0)
                    .then(|| self.0.as_basis_coefficients_slice()[i])
                    .unwrap_or_default()
            })
        })
    }
}

pub(crate) struct IsLastRow<Challenge>(pub(crate) Challenge);

impl<Challenge: Field> IsLastRow<Challenge> {
    #[must_use]
    fn fix_var(&self, z_i: Challenge) -> Self {
        Self(self.0 * z_i)
    }

    #[inline]
    fn eval_packed<Val: Field>(&self) -> Challenge::ExtensionPacking
    where
        Challenge: ExtensionField<Val>,
    {
        Challenge::ExtensionPacking::from_basis_coefficients_fn(|i| {
            Val::Packing::from_fn(|j| {
                (j == Val::Packing::WIDTH - 1)
                    .then(|| self.0.as_basis_coefficients_slice()[i])
                    .unwrap_or_default()
            })
        })
    }
}

pub(crate) struct IsTransition<Challenge>(pub(crate) Challenge);

impl<Challenge: Field> IsTransition<Challenge> {
    #[must_use]
    fn fix_var(&self, z_i: Challenge) -> Self {
        Self((self.0 - Challenge::ONE) * z_i + Challenge::ONE)
    }

    #[inline]
    fn eval_packed<Val: Field>(&self) -> Challenge::ExtensionPacking
    where
        Challenge: ExtensionField<Val>,
    {
        Challenge::ExtensionPacking::from_basis_coefficients_fn(|i| {
            Val::Packing::from_fn(|j| {
                if j == Val::Packing::WIDTH - 1 {
                    self.0.as_basis_coefficients_slice()[i]
                } else if i == 0 {
                    Val::ONE
                } else {
                    Val::ZERO
                }
            })
        })
    }
}
