use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::{chain, cloned};
use p3_air::{Air, BaseAir};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{
    AirMeta, CompressedRoundPoly, EqHelper, ExtensionPacking, FieldSlice, PackedExtensionValue,
    ProverFolderGeneric, ProverFolderOnExtension, ProverFolderOnExtensionPacking,
    ProverFolderOnPacking, RoundPoly, fix_var, vec_add,
};

pub(crate) struct RegularSumcheckProver<'a, Val: Field, Challenge: ExtensionField<Val>, A> {
    pub(crate) meta: AirMeta,
    pub(crate) air: &'a A,
    pub(crate) public_values: &'a [Val],
    pub(crate) claim: Challenge,
    pub(crate) witness: Witness<Val, Challenge>,
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
        let width = meta.width;
        let height = trace.height();
        let log_height = log2_strict_usize(height);
        let log_packing_width = log2_strict_usize(Val::Packing::WIDTH);
        let witness = if log_height > log_packing_width {
            let trace = info_span!("pack trace local and next together").in_scope(|| {
                let len = trace.values.len();
                let packed_len = len >> log_packing_width;
                RowMajorMatrix::new(
                    (0..2 * packed_len)
                        .into_par_iter()
                        .map(|i| {
                            let row = (i / width).div_ceil(2);
                            let col = i % width;
                            Val::Packing::from_fn(|j| {
                                trace.values[(row * width + col + j * packed_len) % len]
                            })
                        })
                        .collect(),
                    2 * width,
                )
            });
            Witness::Packing(trace)
        } else {
            let local = trace.values.par_chunks(width);
            let next = trace.values[width..]
                .par_chunks(width)
                .chain([&trace.values[..width]]);
            let trace = RowMajorMatrix::new(
                local
                    .zip(next)
                    .flat_map(|(local, next)| local.par_iter().chain(next))
                    .copied()
                    .map(Challenge::from)
                    .collect(),
                2 * width,
            );
            Witness::Extension(trace)
        };
        Self {
            air,
            meta,
            public_values,
            claim,
            witness,
            alpha_powers,
            starting_round: max_regular_rounds - log_height,
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

        let round_poly = match &mut self.witness {
            Witness::Packing(..) => self.compute_eq_weighted_round_poly_packing(round, eq_helper),
            Witness::ExtensionPacking(..) => {
                self.compute_eq_weighted_round_poly_extension_packing(round, eq_helper)
            }
            Witness::Extension(..) => {
                self.compute_eq_weighted_round_poly_extension(round, eq_helper)
            }
        };
        self.round_poly = round_poly.clone();
        round_poly.into_compressed()
    }

    #[instrument(skip_all, name = "compute eq weighted round poly (packing)", fields(dim = %self.witness.dim()))]
    fn compute_eq_weighted_round_poly_packing(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let Witness::Packing(trace) = &mut self.witness else {
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
                    let lo = main.row(0);
                    let hi = main.row(1);
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

    #[instrument(skip_all, name = "compute eq weighted round poly (ext packing)", fields(dim = %self.witness.dim()))]
    fn compute_eq_weighted_round_poly_extension_packing(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let Witness::ExtensionPacking(trace) = &mut self.witness else {
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
                    let lo = main.row(0);
                    let hi = main.row(1);
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

    #[instrument(skip_all, name = "compute eq weighted round poly (ext)", fields(dim = %self.witness.dim()))]
    fn compute_eq_weighted_round_poly_extension(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let Witness::Extension(trace) = &mut self.witness else {
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
                    let lo = main.row(0);
                    let hi = main.row(1);
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
        self.witness = match &self.witness {
            Witness::Packing(trace) => {
                let trace = fix_var(trace.as_view(), z_i.into());
                if trace.height() == 1 {
                    Witness::Extension(unpack_row(&trace.values))
                } else {
                    Witness::ExtensionPacking(trace)
                }
            }
            Witness::ExtensionPacking(trace) => {
                let trace = fix_var(trace.as_view(), z_i.into());
                if trace.height() == 1 {
                    Witness::Extension(unpack_row(&trace.values))
                } else {
                    Witness::ExtensionPacking(trace)
                }
            }
            Witness::Extension(trace) => {
                let trace = fix_var(trace.as_view(), z_i);
                Witness::Extension(trace)
            }
        };
        self.is_first_row.fix_var(z_i);
        self.is_last_row.fix_var(z_i);
        self.is_transition.fix_var(z_i);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        let Witness::Extension(trace) = self.witness else {
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
    fn fix_var(&mut self, z_i: Challenge) {
        self.0 *= Challenge::ONE - z_i
    }

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
    fn fix_var(&mut self, z_i: Challenge) {
        self.0 *= z_i
    }

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
    fn fix_var(&mut self, z_i: Challenge) {
        self.0 = (self.0 - Challenge::ONE) * z_i + Challenge::ONE
    }

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

pub(crate) enum Witness<Val: Field, Challenge: ExtensionField<Val>> {
    Packing(RowMajorMatrix<Val::Packing>),
    ExtensionPacking(RowMajorMatrix<Challenge::ExtensionPacking>),
    Extension(RowMajorMatrix<Challenge>),
}

impl<Val: Field, Challenge: ExtensionField<Val>> Witness<Val, Challenge> {
    fn dim(&self) -> usize {
        match self {
            Self::Packing(trace) => {
                log2_strict_usize(trace.height()) + log2_strict_usize(Val::Packing::WIDTH)
            }
            Self::ExtensionPacking(trace) => {
                log2_strict_usize(trace.height()) + log2_strict_usize(Val::Packing::WIDTH)
            }
            Self::Extension(trace) => log2_strict_usize(trace.height()),
        }
    }
}

pub(crate) fn unpack_row<Val: Field, Challenge: ExtensionField<Val>>(
    row: &[Challenge::ExtensionPacking],
) -> RowMajorMatrix<Challenge> {
    let width = row.len();
    RowMajorMatrix::new(
        (0..width * Val::Packing::WIDTH)
            .into_par_iter()
            .map(|i| {
                Challenge::from_basis_coefficients_fn(|j| {
                    row[i % width].as_basis_coefficients_slice()[j].as_slice()[i / width]
                })
            })
            .collect(),
        width,
    )
}
