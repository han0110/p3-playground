use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use core::marker::PhantomData;
use core::mem;

use itertools::{Itertools, chain, cloned, izip};
use p3_air::{Air, BaseAir};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing,
    batch_multiplicative_inverse, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{
    AirMeta, CompressedRoundPoly, EqHelper, ExtensionPacking, FieldSlice, PackedExtensionValue,
    ProverFolderGeneric, ProverFolderOnExtension, ProverFolderOnExtensionPacking,
    ProverFolderOnPacking, RoundPoly, eq_expand, fix_var, unpack_row, vec_add,
};

// TODO: Find a better way to choose the optimal switch-over automatically.
const SWITCH_OVER: usize = 6;

struct EvalState<Val: Field, Challenge: ExtensionField<Val>, Var> {
    width: usize,
    main_eval: Vec<Var>,
    main_diff: Vec<Var>,
    is_first_row_diff: Option<Var>,
    is_first_row_eval: Var,
    is_last_row_diff: Option<Var>,
    is_last_row_eval: Var,
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
            self.is_transition_eval = Var::ONE - self.is_last_row_eval
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

struct IsFirstRow<Challenge>(Challenge);

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

struct IsLastRow<Challenge>(Challenge);

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

struct Quotient<Val, Challenge, VarEF> {
    q_bar: Vec<VarEF>,
    q: RowMajorMatrix<VarEF>,
    barycentric_denoms: Vec<Challenge>,
    _marker: PhantomData<Val>,
}

impl<Val, Challenge, VarEF> Quotient<Val, Challenge, VarEF>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    VarEF: Copy + Send + Sync + PrimeCharacteristicRing + Algebra<Challenge>,
{
    fn new(height: usize, degree: usize) -> Self {
        let barycentric_denoms = batch_multiplicative_inverse(
            &(2..degree as i32 + 1)
                .map(|i| {
                    (0..degree as i32 + 1)
                        .filter(|j| *j != i)
                        .map(|j| Challenge::from_i32(i - j))
                        .product()
                })
                .collect_vec(),
        );
        Self {
            q_bar: Vec::new(),
            q: RowMajorMatrix::new(vec![VarEF::ZERO; (degree - 1) * height / 2], degree - 1),
            barycentric_denoms,
            _marker: PhantomData,
        }
    }

    fn fix_var(&mut self, z_i: Challenge) {
        let weights = izip!(2.., &self.barycentric_denoms)
            .map(|(i, denom)| {
                (0..self.barycentric_denoms.len() + 2)
                    .filter(|j| *j != i)
                    .map(|j| z_i - Challenge::from_usize(j))
                    .product::<Challenge>()
                    * *denom
            })
            .collect_vec();
        if self.q_bar.is_empty() {
            self.q_bar = self
                .q
                .par_rows()
                .map(|row| dot_product(row, cloned(&weights)))
                .collect();
        } else {
            self.q_bar = fix_var(RowMajorMatrixView::new_col(&self.q_bar), z_i.into()).values;
            self.q_bar
                .par_iter_mut()
                .zip(self.q.par_rows())
                .for_each(|(q_bar, row)| {
                    *q_bar += dot_product::<VarEF, _, _>(row, cloned(&weights))
                });
        }
        self.q.values.truncate(self.q.values.len() / 2);
    }
}

impl<Val: Field, Challenge: ExtensionField<Val>>
    Quotient<Val, Challenge, Challenge::ExtensionPacking>
{
    fn eval_1(&self, round: usize, eq_helper: &EqHelper<Val, Challenge>) -> Challenge {
        self.q_bar[1..]
            .par_iter()
            .step_by(2)
            .zip(eq_helper.evals_packed(round))
            .map(|(a, b)| *a * b)
            .sum::<Challenge::ExtensionPacking>()
            .ext_sum()
            * eq_helper.correcting_factor(round)
    }
}

enum Witness<Val: Field, Challenge: ExtensionField<Val>> {
    Packing {
        input: RowMajorMatrix<Val::Packing>,
        quotient: Quotient<Val, Challenge, Challenge::ExtensionPacking>,
        eq_z: Vec<Challenge>,
    },
    ExtensionPacking {
        input: RowMajorMatrix<Challenge::ExtensionPacking>,
        quotient: Quotient<Val, Challenge, Challenge::ExtensionPacking>,
    },
    Extension {
        input: RowMajorMatrix<Challenge>,
    },
}

impl<Val: Field, Challenge: ExtensionField<Val>> Default for Witness<Val, Challenge> {
    fn default() -> Self {
        Self::Extension {
            input: RowMajorMatrix::new(Vec::new(), 0),
        }
    }
}

impl<Val: Field, Challenge: ExtensionField<Val>> Witness<Val, Challenge> {
    fn num_vars(&self) -> usize {
        match self {
            Self::Packing { input, eq_z, .. } => {
                log2_strict_usize(input.height()) + log2_strict_usize(Val::Packing::WIDTH)
                    - log2_strict_usize(eq_z.len())
            }
            Self::ExtensionPacking { input, .. } => {
                log2_strict_usize(input.height()) + log2_strict_usize(Val::Packing::WIDTH)
            }
            Self::Extension { input } => log2_strict_usize(input.height()),
        }
    }
}

pub(crate) struct ZeroSumcheckState<'a, Val: Field, Challenge: ExtensionField<Val>, A> {
    air: &'a A,
    meta: AirMeta,
    public_values: &'a [Val],
    alpha_powers: &'a [Challenge],
    padded_rounds: usize,
    claim: Challenge,
    round_poly: RoundPoly<Challenge>,
    witness: Witness<Val, Challenge>,
    is_first_row: IsFirstRow<Challenge>,
    is_last_row: IsLastRow<Challenge>,
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>, A> ZeroSumcheckState<'a, Val, Challenge, A>
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
        input: RowMajorMatrixView<Val>,
        alpha_powers: &'a [Challenge],
        max_log_height: usize,
    ) -> Self {
        let width = meta.width;
        let height = input.height();
        let log_height = log2_strict_usize(height);
        let log_packing_width = log2_strict_usize(Val::Packing::WIDTH);
        let witness = if log_height > log_packing_width {
            let log_packed_height = log_height - log_packing_width;
            let input = info_span!("pack input local and next together").in_scope(|| {
                let len = input.values.len();
                let packed_len = len >> log_packing_width;
                RowMajorMatrix::new(
                    (0..2 * packed_len)
                        .into_par_iter()
                        .map(|i| {
                            let row = (i / width).div_ceil(2);
                            let col = i % width;
                            Val::Packing::from_fn(|j| {
                                input.values[(row * width + col + j * packed_len) % len]
                            })
                        })
                        .collect(),
                    2 * width,
                )
            });
            let quotient = Quotient::new(1 << log_packed_height, meta.degree);
            let mut eq_z = Vec::with_capacity(1 << min(SWITCH_OVER, log_packed_height));
            eq_z.push(Challenge::ONE);
            Witness::Packing {
                input,
                quotient,
                eq_z,
            }
        } else {
            let local = input.values.par_chunks(width);
            let next = input.values[width..]
                .par_chunks(width)
                .chain([&input.values[..width]]);
            let input = RowMajorMatrix::new(
                local
                    .zip(next)
                    .flat_map(|(local, next)| local.par_iter().chain(next))
                    .copied()
                    .map(Challenge::from)
                    .collect(),
                2 * width,
            );
            Witness::Extension { input }
        };
        Self {
            air,
            meta,
            public_values,
            alpha_powers,
            padded_rounds: max_log_height - log_height,
            claim: Challenge::ZERO,
            round_poly: RoundPoly(Vec::new()),
            witness,
            is_first_row: IsFirstRow(Challenge::ONE),
            is_last_row: IsLastRow(Challenge::ONE),
        }
    }

    pub(crate) fn compute_round_poly(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> CompressedRoundPoly<Challenge> {
        if round < self.padded_rounds {
            return CompressedRoundPoly(vec![Challenge::ZERO]);
        }

        self.round_poly = match &mut self.witness {
            Witness::Packing { eq_z, .. } if eq_z.len() == 1 => {
                self.compute_first_round_poly(round, eq_helper)
            }
            Witness::Packing { .. } => self.compute_round_poly_algo_2(round, eq_helper),
            Witness::ExtensionPacking { .. } => self.compute_round_poly_algo_1(round, eq_helper),
            Witness::Extension { .. } => self.compute_round_poly_algo_1_small(round, eq_helper),
        };
        self.round_poly.clone().into_compressed()
    }

    #[instrument(skip_all, fields(num_vars = %self.witness.num_vars()))]
    fn compute_first_round_poly(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let Witness::Packing {
            input, quotient, ..
        } = &mut self.witness
        else {
            unreachable!()
        };

        let last_row = input.height() / 2 - 1;

        let extra_evals = input
            .par_row_chunks(2)
            .zip(eq_helper.evals_packed(round))
            .zip(quotient.q.par_rows_mut())
            .enumerate()
            .par_fold_reduce(
                || vec![Challenge::ExtensionPacking::ZERO; self.meta.degree - 1],
                |mut sum, (row, ((main, eq_eval), q))| {
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
                    state.is_transition_eval = Val::Packing::ONE;
                    sum.iter_mut().zip(q).for_each(|(sum, q)| {
                        state.next_point();
                        let eval = state.eval(self.air, self.public_values, self.alpha_powers);
                        *q = eval;
                        *sum += eq_eval * eval;
                    });
                    sum
                },
                vec_add,
            )
            .into_iter()
            .map(|eval| eval.ext_sum() * eq_helper.correcting_factor(round));

        RoundPoly::from_evals(chain![[Challenge::ZERO; 2], extra_evals])
    }

    #[instrument(skip_all, fields(num_vars = %self.witness.num_vars()))]
    fn compute_round_poly_algo_2(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let Witness::Packing {
            input,
            quotient,
            eq_z,
        } = &mut self.witness
        else {
            unreachable!()
        };

        let last_row = input.height() / (2 * eq_z.len()) - 1;

        let eval_1 = quotient.eval_1(round, eq_helper);
        let eval_0 = eq_helper.eval_0(round, self.claim, eval_1);
        let extra_evals = input
            .par_row_chunks(2 * eq_z.len())
            .zip(eq_helper.evals_packed(round))
            .zip(quotient.q_bar.par_chunks(2))
            .zip(quotient.q.par_rows_mut())
            .enumerate()
            .par_fold_reduce(
                || vec![Challenge::ExtensionPacking::ZERO; self.meta.degree - 1],
                |mut sum, (row, (((main, eq_eval), q_bar), q))| {
                    let mut state = EvalState::new(self.meta.width);
                    eq_z.iter().enumerate().for_each(|(i, eq_z_i)| {
                        state.main_diff.slice_sub_assign_scaled_iter(
                            main.row(i),
                            Challenge::ExtensionPacking::from(*eq_z_i),
                        );
                    });
                    eq_z.iter().enumerate().for_each(|(i, eq_z_i)| {
                        state.main_eval.slice_add_assign_scaled_iter(
                            main.row(eq_z.len() + i),
                            Challenge::ExtensionPacking::from(*eq_z_i),
                        );
                    });
                    state.main_diff.slice_add_assign(&state.main_eval);
                    state.is_first_row_eval = Challenge::ExtensionPacking::ZERO;
                    state.is_first_row_diff = (row == 0).then(|| -self.is_first_row.eval_packed());
                    state.is_last_row_diff =
                        (row == last_row).then(|| self.is_last_row.eval_packed());
                    state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                    state.is_transition_eval =
                        Challenge::ExtensionPacking::ONE - state.is_last_row_eval;
                    let mut q_bar_eval = q_bar[1];
                    let q_bar_diff = q_bar[1] - q_bar[0];
                    sum.iter_mut().zip(q).for_each(|(sum, q)| {
                        q_bar_eval += q_bar_diff;
                        state.next_point();
                        let eval =
                            state.eval_packed(self.air, self.public_values, self.alpha_powers);
                        *q = eval - q_bar_eval;
                        *sum += eq_eval * eval;
                    });
                    sum
                },
                vec_add,
            )
            .into_iter()
            .map(|eval| eval.ext_sum() * eq_helper.correcting_factor(round));

        RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
    }

    #[instrument(skip_all, fields(num_vars = %self.witness.num_vars()))]
    fn compute_round_poly_algo_1(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let Witness::ExtensionPacking { input, quotient } = &mut self.witness else {
            unreachable!()
        };

        let last_row = input.height() / 2 - 1;

        let eval_1 = quotient.eval_1(round, eq_helper);
        let eval_0 = eq_helper.eval_0(round, self.claim, eval_1);
        let extra_evals = input
            .par_row_chunks(2)
            .zip(eq_helper.evals_packed(round))
            .zip(quotient.q_bar.par_chunks(2))
            .zip(quotient.q.par_rows_mut())
            .enumerate()
            .par_fold_reduce(
                || vec![Challenge::ExtensionPacking::ZERO; self.meta.degree - 1],
                |mut sum, (row, (((main, eq_eval), q_bar), q))| {
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
                    state.is_transition_eval =
                        Challenge::ExtensionPacking::ONE - state.is_last_row_eval;
                    let mut q_bar_eval = q_bar[1];
                    let q_bar_diff = q_bar[1] - q_bar[0];
                    sum.iter_mut().zip(q).for_each(|(sum, q)| {
                        q_bar_eval += q_bar_diff;
                        state.next_point();
                        let eval =
                            state.eval_packed(self.air, self.public_values, self.alpha_powers);
                        *q = eval - q_bar_eval;
                        *sum += eq_eval * eval;
                    });
                    sum
                },
                vec_add,
            )
            .into_iter()
            .map(|eval| eval.ext_sum() * eq_helper.correcting_factor(round));

        RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
    }

    #[instrument(skip_all, fields(num_vars = %self.witness.num_vars()))]
    fn compute_round_poly_algo_1_small(
        &mut self,
        round: usize,
        eq_helper: &EqHelper<Val, Challenge>,
    ) -> RoundPoly<Challenge> {
        let Witness::Extension { input } = &mut self.witness else {
            unreachable!()
        };

        let last_row = input.height() / 2 - 1;

        let mut extra_evals = input
            .par_row_chunks(2)
            .zip(eq_helper.evals(round))
            .enumerate()
            .par_fold_reduce(
                || vec![Challenge::ZERO; self.meta.degree],
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
                    state.is_transition_eval = Challenge::ONE - state.is_last_row_eval;

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
        self.claim = self.round_poly.subclaim(z_i);

        if round < self.padded_rounds {
            return;
        }

        self.witness = match mem::take(&mut self.witness) {
            Witness::Packing {
                input,
                mut quotient,
                mut eq_z,
            } => {
                quotient.fix_var(z_i);
                eq_z.resize(2 * eq_z.len(), Challenge::ZERO);
                eq_expand(&mut eq_z, z_i, round - self.padded_rounds);
                if eq_z.len() != eq_z.capacity() {
                    Witness::Packing {
                        input,
                        quotient,
                        eq_z,
                    }
                } else {
                    let input = info_span!("switch over").in_scope(|| {
                        let len = input.values.len() / eq_z.len();
                        let mut values = RowMajorMatrix::new(
                            Challenge::ExtensionPacking::zero_vec(len),
                            input.width(),
                        );
                        values.par_rows_mut().enumerate().for_each(|(i, row)| {
                            eq_z.iter().enumerate().for_each(|(j, eq_z_j)| {
                                row.slice_add_assign_scaled_iter(
                                    input.row(i * eq_z.len() + j),
                                    Challenge::ExtensionPacking::from(*eq_z_j),
                                );
                            });
                        });
                        values
                    });
                    if input.height() == 1 {
                        Witness::Extension {
                            input: unpack_row(&input.values),
                        }
                    } else {
                        Witness::ExtensionPacking { input, quotient }
                    }
                }
            }
            Witness::ExtensionPacking {
                mut input,
                mut quotient,
            } => {
                input = fix_var(input.as_view(), z_i.into());
                quotient.fix_var(z_i);
                if input.height() == 1 {
                    Witness::Extension {
                        input: unpack_row(&input.values),
                    }
                } else {
                    Witness::ExtensionPacking { input, quotient }
                }
            }
            Witness::Extension { mut input } => {
                input = fix_var(input.as_view(), z_i);
                Witness::Extension { input }
            }
        };
        self.is_first_row.fix_var(z_i);
        self.is_last_row.fix_var(z_i);
    }

    pub(crate) fn into_evals(self) -> Vec<Challenge> {
        let Witness::Extension { input, .. } = self.witness else {
            unreachable!()
        };
        input.values
    }
}
