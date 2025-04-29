use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::{chain, cloned};
use p3_air::{Air, BaseAir};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    AirMeta, AirTrace, CompressedRoundPoly, EqHelper, ExtensionPacking, FieldSlice,
    PackedExtensionValue, ProverFolderGeneric, ProverFolderOnExtension,
    ProverFolderOnExtensionPacking, ProverFolderOnPacking, RoundPoly, vec_add,
};

pub(crate) struct RegularSumcheckProver<'a, Val: Field, Challenge: ExtensionField<Val>, A> {
    pub(crate) meta: AirMeta,
    pub(crate) air: &'a A,
    pub(crate) public_values: &'a [Val],
    pub(crate) claim: Challenge,
    pub(crate) trace: AirTrace<Val, Challenge>,
    pub(crate) alpha_powers: &'a [Challenge],
    pub(crate) starting_round: usize,
    pub(crate) is_first_row: IsFirstRow<Challenge>,
    pub(crate) is_last_row: IsLastRow<Challenge>,
    pub(crate) is_transition: IsTransition<Challenge>,
    pub(crate) round_poly: RoundPoly<Challenge>,
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>, A> RegularSumcheckProver<'a, Val, Challenge, A>
where
    A: BaseAir<Val>
        + for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    pub(crate) fn new(
        meta: AirMeta,
        air: &'a A,
        public_values: &'a [Val],
        claim: Challenge,
        trace: RowMajorMatrixView<Val>,
        alpha_powers: &'a [Challenge],
        max_regular_rounds: usize,
    ) -> Self {
        let starting_round = max_regular_rounds - log2_strict_usize(trace.height());
        Self {
            air,
            meta,
            public_values,
            claim,
            trace: AirTrace::new(trace),
            alpha_powers,
            starting_round,
            is_first_row: IsFirstRow(Challenge::ONE),
            is_last_row: IsLastRow(Challenge::ONE),
            is_transition: IsTransition(Challenge::ZERO),
            round_poly: Default::default(),
        }
    }

    pub(crate) fn compute_eq_weighted_round_poly(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> CompressedRoundPoly<Challenge> {
        if round < self.starting_round {
            return CompressedRoundPoly::default();
        }

        let round_poly = match &mut self.trace {
            AirTrace::Packing(_) => self.compute_eq_weighted_round_poly_packing(round, eq_helper),
            AirTrace::ExtensionPacking(_) => {
                self.compute_eq_weighted_round_poly_extension_packing(round, eq_helper)
            }
            AirTrace::Extension(_) => {
                self.compute_eq_weighted_round_poly_extension(round, eq_helper)
            }
        };
        self.round_poly = round_poly.clone();
        round_poly.into_compressed()
    }

    #[instrument(skip_all, name = "compute eq weighted round poly (packing)", fields(log_h = %self.trace.log_height()))]
    fn compute_eq_weighted_round_poly_packing(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let AirTrace::Packing(trace) = &mut self.trace else {
            unreachable!()
        };

        let last_row = trace.height() / 2 - 1;

        let mut extra_evals = trace
            .par_row_chunks(2)
            .zip(eq_helper.evals_packed(round))
            .enumerate()
            .par_fold_reduce(
                || vec![Challenge::ExtensionPacking::ZERO; self.meta.multivariate_degree],
                |mut sum, (row, (main, eq_eval))| {
                    let lo = main.row(0).unwrap();
                    let hi = main.row(1).unwrap();
                    let mut state = EvalState::new(self.meta.width);
                    state.main_eval.slice_assign_iter(hi);
                    state.main_diff.slice_sub_iter(cloned(&state.main_eval), lo);
                    state.is_first_row_eval = Val::Packing::ZERO;
                    state.is_first_row_diff =
                        (row == 0).then(|| -IsFirstRow(Val::ONE).eval_packed());
                    state.is_last_row_diff =
                        (row == last_row).then(|| IsLastRow(Val::ONE).eval_packed());
                    state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                    state.is_transition_eval = if row == last_row {
                        IsTransition(Val::ZERO).eval_packed()
                    } else {
                        Val::Packing::ONE
                    };
                    state.is_transition_diff =
                        (row == last_row).then(|| state.is_transition_eval - Val::Packing::ONE);
                    sum.iter_mut().enumerate().for_each(|(d, sum)| {
                        if d > 0 {
                            state.next_point();
                        }
                        let eval = state.eval(self.air, self.public_values, self.alpha_powers);
                        *sum += eq_eval * eval;
                    });
                    sum
                },
                vec_add,
            )
            .into_iter()
            .map(|eval| eval.ext_sum() * eq_helper.correcting_factor(round));
        let eval_1 = extra_evals.next().unwrap();
        let eval_0 = eq_helper.eval_0(round, self.claim, eval_1);

        RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
    }

    #[instrument(skip_all, name = "compute eq weighted round poly (ext packing)", fields(log_h = %self.trace.log_height()))]
    fn compute_eq_weighted_round_poly_extension_packing(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let AirTrace::ExtensionPacking(trace) = &mut self.trace else {
            unreachable!()
        };

        let last_row = trace.height() / 2 - 1;

        let mut extra_evals = trace
            .par_row_chunks(2)
            .zip(eq_helper.evals_packed(round))
            .enumerate()
            .par_fold_reduce(
                || vec![Challenge::ExtensionPacking::ZERO; self.meta.multivariate_degree],
                |mut sum, (row, (main, eq_eval))| {
                    let lo = main.row(0).unwrap();
                    let hi = main.row(1).unwrap();
                    let mut state = EvalState::new(self.meta.width);
                    state.main_eval.slice_assign_iter(hi);
                    state.main_diff.slice_sub_iter(cloned(&state.main_eval), lo);
                    state.is_first_row_eval = Challenge::ExtensionPacking::ZERO;
                    state.is_first_row_diff = (row == 0).then(|| -self.is_first_row.eval_packed());
                    state.is_last_row_diff =
                        (row == last_row).then(|| self.is_last_row.eval_packed());
                    state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                    state.is_transition_eval = if row == last_row {
                        self.is_transition.eval_packed()
                    } else {
                        Challenge::ExtensionPacking::ONE
                    };
                    state.is_transition_diff = (row == last_row)
                        .then(|| state.is_transition_eval - Challenge::ExtensionPacking::ONE);
                    sum.iter_mut().enumerate().for_each(|(d, sum)| {
                        if d > 0 {
                            state.next_point();
                        }
                        let eval =
                            state.eval_packed(self.air, self.public_values, self.alpha_powers);
                        *sum += eq_eval * eval;
                    });
                    sum
                },
                vec_add,
            )
            .into_iter()
            .map(|eval| eval.ext_sum() * eq_helper.correcting_factor(round));
        let eval_1 = extra_evals.next().unwrap();
        let eval_0 = eq_helper.eval_0(round, self.claim, eval_1);

        RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
    }

    #[instrument(skip_all, name = "compute eq weighted round poly (ext)", fields(log_h = %self.trace.log_height()))]
    fn compute_eq_weighted_round_poly_extension(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let AirTrace::Extension(trace) = &mut self.trace else {
            unreachable!()
        };

        let last_row = trace.height() / 2 - 1;

        let mut extra_evals = trace
            .par_row_chunks(2)
            .zip(eq_helper.evals(round))
            .enumerate()
            .par_fold_reduce(
                || vec![Challenge::ZERO; self.meta.multivariate_degree],
                |mut sum, (row, (main, eq_eval))| {
                    let lo = main.row(0).unwrap();
                    let hi = main.row(1).unwrap();
                    let mut state = EvalState::new(self.meta.width);
                    state.main_eval.slice_assign_iter(hi);
                    state.main_diff.slice_sub_iter(cloned(&state.main_eval), lo);
                    state.is_first_row_eval = Challenge::ZERO;
                    state.is_first_row_diff = (row == 0).then(|| -self.is_first_row.0);
                    state.is_last_row_diff = (row == last_row).then_some(self.is_last_row.0);
                    state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                    state.is_transition_eval = if row == last_row {
                        self.is_transition.0
                    } else {
                        Challenge::ONE
                    };
                    state.is_transition_diff =
                        (row == last_row).then(|| state.is_transition_eval - Challenge::ONE);
                    sum.iter_mut().enumerate().for_each(|(d, sum)| {
                        if d > 0 {
                            state.next_point();
                        }
                        let eval = state.eval(self.air, self.public_values, self.alpha_powers);
                        *sum += eq_eval * eval;
                    });
                    sum
                },
                vec_add,
            )
            .into_iter()
            .map(|eval| eval * eq_helper.correcting_factor(round));
        let eval_1 = extra_evals.next().unwrap();
        let eval_0 = eq_helper.eval_0(round, self.claim, eval_1);

        RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
    }

    pub(crate) fn fix_var(&mut self, round: usize, z_i: Challenge) {
        if round < self.starting_round {
            return;
        }

        self.claim = self.round_poly.subclaim(z_i);
        self.trace = self.trace.fix_var(z_i);
        self.is_first_row = self.is_first_row.fix_var(z_i);
        self.is_last_row = self.is_last_row.fix_var(z_i);
        self.is_transition = self.is_transition.fix_var(z_i);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        let AirTrace::Extension(trace) = self.trace else {
            unreachable!()
        };
        trace.values
    }
}

struct EvalState<Val: Field, Challenge: ExtensionField<Val>, Var> {
    width: usize,
    main_eval: Vec<Var>,
    main_diff: Vec<Var>,
    is_first_row_diff: Option<Var>,
    is_first_row_eval: Var,
    is_last_row_diff: Option<Var>,
    is_last_row_eval: Var,
    is_transition_diff: Option<Var>,
    is_transition_eval: Var,
    _marker: PhantomData<(Val, Challenge)>,
}

impl<Val: Field, Challenge: ExtensionField<Val>, Var: Copy + Send + Sync + PrimeCharacteristicRing>
    EvalState<Val, Challenge, Var>
{
    #[inline]
    fn new(width: usize) -> Self {
        Self {
            width,
            main_eval: vec![Var::ZERO; 2 * width],
            main_diff: vec![Var::ZERO; 2 * width],
            is_first_row_diff: None,
            is_first_row_eval: Var::ZERO,
            is_last_row_diff: None,
            is_last_row_eval: Var::ZERO,
            is_transition_diff: None,
            is_transition_eval: Var::ZERO,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn next_point(&mut self) {
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
    fn eval<A, VarEF>(&self, air: &A, public_values: &[Val], alpha_powers: &[Challenge]) -> VarEF
    where
        A: for<'t> Air<ProverFolderGeneric<'t, Val, Challenge, Var, VarEF>>,
        Var: Algebra<Val>,
        VarEF: Algebra<Var> + From<Challenge>,
    {
        let mut builder = ProverFolderGeneric {
            main: DenseMatrix::new(&self.main_eval, self.width),
            public_values,
            is_first_row: self.is_first_row_eval,
            is_last_row: self.is_last_row_eval,
            is_transition: self.is_transition_eval,
            alpha_powers,
            accumulator: VarEF::ZERO,
            constraint_index: 0,
        };
        air.eval(&mut builder);
        builder.accumulator
    }
}

impl<Val: Field, Challenge: ExtensionField<Val>>
    EvalState<Val, Challenge, Challenge::ExtensionPacking>
{
    #[inline]
    fn eval_packed<A>(
        &self,
        air: &A,
        public_values: &[Val],
        alpha_powers: &[Challenge],
    ) -> Challenge::ExtensionPacking
    where
        A: for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
    {
        let mut builder = ProverFolderOnExtensionPacking {
            main: DenseMatrix::new(ExtensionPacking::from_slice(&self.main_eval), self.width),
            public_values,
            is_first_row: ExtensionPacking(self.is_first_row_eval),
            is_last_row: ExtensionPacking(self.is_last_row_eval),
            is_transition: ExtensionPacking(self.is_transition_eval),
            alpha_powers,
            accumulator: ExtensionPacking(Challenge::ExtensionPacking::ZERO),
            constraint_index: 0,
        };
        air.eval(&mut builder);
        builder.accumulator.0
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
