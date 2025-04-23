use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use core::marker::PhantomData;

use itertools::{Itertools, chain, cloned, izip};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::FieldChallenger;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing, batch_multiplicative_inverse, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{
    CompressedRoundPoly, ExtensionPacking, FieldSlice, PackedExtensionValue, Proof,
    ProverFolderWithExtensionPacking, ProverFolderWithVal, SumcheckProof, SymbolicAirBuilder,
    eq_expand, eq_poly, fix_var, horner, meta, vander_mat_inv, vec_add,
};

#[instrument(skip_all)]
pub fn prove<Val, Challenge, A>(
    air: &A,
    public_values: &[Val],
    input: RowMajorMatrix<Val>,
    mut challenger: impl FieldChallenger<Val>,
) -> Proof<Challenge>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<ProverFolderWithVal<'t, Val, Challenge, Challenge, Challenge>>
        + for<'t> Air<
            ProverFolderWithVal<'t, Val, Challenge, Val::Packing, Challenge::ExtensionPacking>,
        > + for<'t> Air<ProverFolderWithExtensionPacking<'t, Val, Challenge>>,
{
    assert_eq!(air.width(), input.width());

    // TODO: Preprocess the meta.
    let (constraint_count, degree) = meta(air);

    // TODO: PCS commit and observe.

    challenger.observe_slice(public_values);

    // TODO: Prove LogUp with fractional sumchecks.

    let (_z, zero_sumcheck) = prove_zero_sumcheck(
        air,
        constraint_count,
        degree,
        public_values,
        input.as_view(),
        &mut challenger,
    );

    // TODO: PCS open

    // TODO: Remove the following sanity checks.
    #[cfg(debug_assertions)]
    {
        let (local, next) = zero_sumcheck.evals.split_at(air.width());
        let mut eq_z = eq_poly(&_z, Challenge::ONE);
        assert_eq!(input.columnwise_dot_product(&eq_z), local);
        eq_z.rotate_right(1);
        assert_eq!(input.columnwise_dot_product(&eq_z), next);
    }

    Proof { zero_sumcheck }
}

struct RoundPoly<Challenge>(Vec<Challenge>);

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

    fn subclaim(&self, z_i: Challenge) -> Challenge {
        horner(&self.0, z_i)
    }

    fn iter_compressed(&self) -> impl Iterator<Item = Challenge> {
        chain![[self.0[0]], self.0[2..].iter().copied()]
    }

    fn into_compressed(mut self) -> CompressedRoundPoly<Challenge> {
        self.0.remove(1);
        CompressedRoundPoly(self.0)
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

struct EqHelper<'a, Val, Challenge, VarEF> {
    evals: Vec<VarEF>,
    round: usize,
    r: &'a [Challenge],
    one_minus_r_inv: Vec<Challenge>,
    correcting_factor: Challenge,
    _marker: PhantomData<Val>,
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>, VarEF: Send + Sync>
    EqHelper<'a, Val, Challenge, VarEF>
{
    fn new(r: &'a [Challenge], scalar: VarEF) -> Self
    where
        VarEF: Copy + Algebra<Challenge>,
    {
        let evals = eq_poly(&r[1..], scalar);
        Self {
            evals,
            round: 0,
            r,
            one_minus_r_inv: batch_multiplicative_inverse(
                &r.iter().map(|r_i| Challenge::ONE - *r_i).collect_vec(),
            ),
            correcting_factor: Challenge::ONE,
            _marker: PhantomData,
        }
    }

    fn evals(&self) -> impl IndexedParallelIterator<Item = &VarEF> {
        self.evals.par_iter().step_by(1 << self.round)
    }

    fn next_round(&mut self) {
        self.round += 1;
        if let Some(one_minus_r_i_inv) = self.one_minus_r_inv.get(self.round) {
            self.correcting_factor *= *one_minus_r_i_inv;
        }
    }

    fn eval_0(&self, claim: Challenge, eval_1: Challenge) -> Challenge {
        (claim - self.r[self.round] * eval_1) * self.one_minus_r_inv[self.round]
    }
}

impl<'a, Val: Field, Challenge: ExtensionField<Val>>
    EqHelper<'a, Val, Challenge, Challenge::ExtensionPacking>
{
    fn new_packed(r: &'a [Challenge]) -> Self {
        let log_packing_width = log2_strict_usize(Val::Packing::WIDTH);
        let (r_lo, r_hi) = r.split_at(r.len() - log_packing_width);
        let scalar = Challenge::ExtensionPacking::from_ext_slice(&eq_poly(r_hi, Challenge::ONE));
        Self::new(r_lo, scalar)
    }
}

struct Quotient<Val, Challenge, VarEF> {
    q_bar: Vec<VarEF>,
    q: RowMajorMatrix<VarEF>,
    barycentric_denoms: Vec<Challenge>,
    _marker: PhantomData<Val>,
}

impl<
    Val: Field,
    Challenge: ExtensionField<Val>,
    VarEF: Copy + Send + Sync + PrimeCharacteristicRing + Algebra<Challenge>,
> Quotient<Val, Challenge, VarEF>
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

    fn eval_1(&self, eq_helper: &EqHelper<Val, Challenge, VarEF>) -> VarEF {
        self.q_bar[1..]
            .par_iter()
            .step_by(2)
            .zip(eq_helper.evals())
            .map(|(a, b)| *a * *b)
            .sum::<VarEF>()
            * eq_helper.correcting_factor
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
        A: for<'t> Air<ProverFolderWithVal<'t, Val, Challenge, Var, VarEF>>,
        Var: Algebra<Val>,
        VarEF: Algebra<Var> + From<Challenge>,
    {
        let mut builder = ProverFolderWithVal {
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
        A: for<'t> Air<ProverFolderWithExtensionPacking<'t, Val, Challenge>>,
    {
        let mut builder = ProverFolderWithExtensionPacking {
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

pub fn prove_zero_sumcheck<Val, Challenge, A>(
    air: &A,
    constraint_count: usize,
    degree: usize,
    public_values: &[Val],
    input: RowMajorMatrixView<Val>,
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<Challenge>, SumcheckProof<Challenge>)
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<ProverFolderWithVal<'t, Val, Challenge, Challenge, Challenge>>
        + for<'t> Air<
            ProverFolderWithVal<'t, Val, Challenge, Val::Packing, Challenge::ExtensionPacking>,
        > + for<'t> Air<ProverFolderWithExtensionPacking<'t, Val, Challenge>>,
{
    let width = air.width();
    let log_height = log2_strict_usize(input.height());
    let log_packing_width = log2_strict_usize(Val::Packing::WIDTH);

    if log_height == 0 {
        return (
            Vec::new(),
            SumcheckProof {
                num_vars: 0,
                compressed_round_polys: Vec::new(),
                evals: chain![input.values, input.values]
                    .copied()
                    .map(Challenge::from)
                    .collect(),
            },
        );
    }

    challenger.observe(Val::from_u8(log_height as u8));

    let r = (0..log_height)
        .map(|_| challenger.sample_algebra_element())
        .collect_vec();

    let alpha = challenger.sample_algebra_element::<Challenge>();
    let mut alpha_powers = alpha.powers().take(constraint_count).collect_vec();
    alpha_powers.reverse();

    let mut claim = Challenge::ZERO;
    let mut z = Vec::with_capacity(log_height);
    let mut compressed_round_polys = Vec::with_capacity(log_height);
    let mut is_first_row = IsFirstRow(Challenge::ONE);
    let mut is_last_row = IsLastRow(Challenge::ONE);

    let mut input = if log_height > log_packing_width {
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

        let mut eq_helper = EqHelper::new_packed(&r);
        let mut quotient = Quotient::new(input.height(), degree);

        {
            let round_poly = compute_first_round_poly(
                air,
                degree,
                public_values,
                &input,
                &alpha_powers,
                &eq_helper,
                &mut quotient,
            );

            round_poly
                .iter_compressed()
                .for_each(|coeff| challenger.observe_algebra_element(coeff));
            let z_i = challenger.sample_algebra_element();

            claim = round_poly.subclaim(z_i);
            is_first_row.fix_var(z_i);
            is_last_row.fix_var(z_i);
            eq_helper.next_round();
            quotient.fix_var(z_i);

            compressed_round_polys.push(round_poly.into_compressed());
            z.push(z_i);
        }

        let log_packing_height = log_height - log_packing_width;

        // TODO: Find a better way to choose the optimal switch-over automatically.
        let switch_over = min(6, log_packing_height);

        let mut eq_z = Challenge::zero_vec(1 << switch_over);
        eq_z[0] = is_first_row.0;
        eq_z[1] = is_last_row.0;

        for round in 1..switch_over {
            let round_poly = compute_round_poly_algo_2(
                air,
                degree,
                claim,
                public_values,
                &input,
                &alpha_powers,
                &is_first_row,
                &is_last_row,
                &eq_helper,
                &eq_z[..1 << round],
                &mut quotient,
            );

            round_poly
                .iter_compressed()
                .for_each(|coeff| challenger.observe_algebra_element(coeff));
            let z_i = challenger.sample_algebra_element();

            claim = round_poly.subclaim(z_i);
            is_first_row.fix_var(z_i);
            is_last_row.fix_var(z_i);
            eq_helper.next_round();
            eq_expand(&mut eq_z, z_i, round);
            quotient.fix_var(z_i);

            compressed_round_polys.push(round_poly.into_compressed());
            z.push(z_i);
        }

        let mut input = info_span!("switch over").in_scope(|| {
            let len = input.values.len() / eq_z.len();
            let mut values =
                RowMajorMatrix::new(Challenge::ExtensionPacking::zero_vec(len), input.width());
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

        drop(eq_z);

        for _ in switch_over..log_packing_height {
            let round_poly = compute_round_poly_algo_1_packed(
                air,
                degree,
                claim,
                public_values,
                &input,
                &alpha_powers,
                &is_first_row,
                &is_last_row,
                &eq_helper,
                &mut quotient,
            );

            round_poly
                .iter_compressed()
                .for_each(|coeff| challenger.observe_algebra_element(coeff));
            let z_i = challenger.sample_algebra_element();

            claim = round_poly.subclaim(z_i);
            is_first_row.fix_var(z_i);
            is_last_row.fix_var(z_i);
            eq_helper.next_round();
            input = fix_var(input.as_view(), z_i.into());
            quotient.fix_var(z_i);

            compressed_round_polys.push(round_poly.into_compressed());
            z.push(z_i);
        }

        RowMajorMatrix::new(
            (0..1 << log_packing_width)
                .into_par_iter()
                .flat_map(|i| {
                    input.values.par_iter().map(move |v| {
                        Challenge::from_basis_coefficients_iter(
                            v.as_basis_coefficients_slice()
                                .iter()
                                .map(|p| p.as_slice()[i]),
                        )
                    })
                })
                .collect(),
            input.width(),
        )
    } else {
        let local = input.values.par_chunks(width);
        let next = input.values[width..]
            .par_chunks(width)
            .chain([&input.values[..width]]);
        RowMajorMatrix::new(
            local
                .zip(next)
                .flat_map(|(local, next)| local.par_iter().chain(next))
                .copied()
                .map(Challenge::from)
                .collect(),
            2 * width,
        )
    };

    if input.height() > 1 {
        let round = r.len() - log2_strict_usize(input.height());

        let mut eq_helper = EqHelper::new(&r[round..], Challenge::ONE);

        for _ in round..log_height {
            let round_poly = compute_round_poly_algo_1(
                air,
                degree,
                claim,
                public_values,
                &input,
                &alpha_powers,
                &is_first_row,
                &is_last_row,
                &eq_helper,
            );

            round_poly
                .iter_compressed()
                .for_each(|coeff| challenger.observe_algebra_element(coeff));
            let z_i = challenger.sample_algebra_element();

            claim = round_poly.subclaim(z_i);
            is_first_row.fix_var(z_i);
            is_last_row.fix_var(z_i);
            eq_helper.next_round();
            input = fix_var(input.as_view(), z_i);

            compressed_round_polys.push(round_poly.into_compressed());
            z.push(z_i);
        }
    };

    let evals = input.values;

    (
        z,
        SumcheckProof {
            num_vars: log_height,
            compressed_round_polys,
            evals,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, fields(dim = %input.height().ilog2()))]
fn compute_first_round_poly<Val, Challenge, A>(
    air: &A,
    degree: usize,
    public_values: &[Val],
    input: &RowMajorMatrix<Val::Packing>,
    alpha_powers: &[Challenge],
    eq_helper: &EqHelper<Val, Challenge, Challenge::ExtensionPacking>,
    quotient: &mut Quotient<Val, Challenge, Challenge::ExtensionPacking>,
) -> RoundPoly<Challenge>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<
        ProverFolderWithVal<'t, Val, Challenge, Val::Packing, Challenge::ExtensionPacking>,
    >,
{
    let width = air.width();
    let last_row = input.height() / 2 - 1;

    let extra_evals = input
        .par_row_chunks(2)
        .zip(eq_helper.evals())
        .zip(quotient.q.par_rows_mut())
        .enumerate()
        .par_fold_reduce(
            || vec![Challenge::ExtensionPacking::ZERO; degree - 1],
            |mut sum, (row, ((main, eq_eval), q))| {
                let lo = main.row(0);
                let hi = main.row(1);
                let mut state = EvalState::new(width);
                state.main_eval.slice_assign_iter(hi);
                state.main_diff.slice_sub_iter(cloned(&state.main_eval), lo);
                state.is_first_row_eval = Val::Packing::ZERO;
                state.is_first_row_diff = (row == 0).then(|| -IsFirstRow(Val::ONE).eval_packed());
                state.is_last_row_diff =
                    (row == last_row).then(|| IsLastRow(Val::ONE).eval_packed());
                state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                state.is_transition_eval = Val::Packing::ONE;
                sum.iter_mut().zip(q).for_each(|(sum, q)| {
                    state.next_point();
                    let eval = state.eval(air, public_values, alpha_powers);
                    *q = eval;
                    *sum += *eq_eval * eval;
                });
                sum
            },
            vec_add,
        )
        .into_iter()
        .map(|eval| eval.ext_sum());

    RoundPoly::from_evals(chain![[Challenge::ZERO; 2], extra_evals])
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, fields(dim = %(input.height() * Val::Packing::WIDTH / eq_z.len()).ilog2()))]
fn compute_round_poly_algo_2<Val, Challenge, A>(
    air: &A,
    degree: usize,
    claim: Challenge,
    public_values: &[Val],
    input: &RowMajorMatrix<Val::Packing>,
    alpha_powers: &[Challenge],
    is_first_row: &IsFirstRow<Challenge>,
    is_last_row: &IsLastRow<Challenge>,
    eq_helper: &EqHelper<Val, Challenge, Challenge::ExtensionPacking>,
    eq_z: &[Challenge],
    quotient: &mut Quotient<Val, Challenge, Challenge::ExtensionPacking>,
) -> RoundPoly<Challenge>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderWithExtensionPacking<'t, Val, Challenge>>,
{
    let width = air.width();
    let last_row = input.height() / (2 * eq_z.len()) - 1;

    let eval_1 = quotient.eval_1(eq_helper).ext_sum();
    let eval_0 = eq_helper.eval_0(claim, eval_1);
    let extra_evals = input
        .par_row_chunks(2 * eq_z.len())
        .zip(eq_helper.evals())
        .zip(quotient.q_bar.par_chunks(2))
        .zip(quotient.q.par_rows_mut())
        .enumerate()
        .par_fold_reduce(
            || vec![Challenge::ExtensionPacking::ZERO; degree - 1],
            |mut sum, (row, (((main, eq_eval), q_bar), q))| {
                let mut state = EvalState::new(width);
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
                state.is_first_row_diff = (row == 0).then(|| -is_first_row.eval_packed());
                state.is_last_row_diff = (row == last_row).then(|| is_last_row.eval_packed());
                state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                state.is_transition_eval =
                    Challenge::ExtensionPacking::ONE - state.is_last_row_eval;
                let mut q_bar_eval = q_bar[1];
                let q_bar_diff = q_bar[1] - q_bar[0];
                sum.iter_mut().zip(q).for_each(|(sum, q)| {
                    q_bar_eval += q_bar_diff;
                    state.next_point();
                    let eval = state.eval_packed(air, public_values, alpha_powers);
                    *q = eval - q_bar_eval;
                    *sum += *eq_eval * eval;
                });
                sum
            },
            vec_add,
        )
        .into_iter()
        .map(|eval| eval.ext_sum() * eq_helper.correcting_factor);

    RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, fields(dim = %input.height().ilog2()))]
fn compute_round_poly_algo_1_packed<Val, Challenge, A>(
    air: &A,
    degree: usize,
    claim: Challenge,
    public_values: &[Val],
    input: &RowMajorMatrix<Challenge::ExtensionPacking>,
    alpha_powers: &[Challenge],
    is_first_row: &IsFirstRow<Challenge>,
    is_last_row: &IsLastRow<Challenge>,
    eq_helper: &EqHelper<Val, Challenge, Challenge::ExtensionPacking>,
    quotient: &mut Quotient<Val, Challenge, Challenge::ExtensionPacking>,
) -> RoundPoly<Challenge>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderWithExtensionPacking<'t, Val, Challenge>>,
{
    let width = air.width();
    let last_row = input.height() / 2 - 1;

    let eval_1 = quotient.eval_1(eq_helper).ext_sum();
    let eval_0 = eq_helper.eval_0(claim, eval_1);
    let extra_evals = input
        .par_row_chunks(2)
        .zip(eq_helper.evals())
        .zip(quotient.q_bar.par_chunks(2))
        .zip(quotient.q.par_rows_mut())
        .enumerate()
        .par_fold_reduce(
            || vec![Challenge::ExtensionPacking::ZERO; degree - 1],
            |mut sum, (row, (((main, eq_eval), q_bar), q))| {
                let lo = main.row(0);
                let hi = main.row(1);
                let mut state = EvalState::new(width);
                state.main_eval.slice_assign_iter(hi);
                state.main_diff.slice_sub_iter(cloned(&state.main_eval), lo);
                state.is_first_row_eval = Challenge::ExtensionPacking::ZERO;
                state.is_first_row_diff = (row == 0).then(|| -is_first_row.eval_packed());
                state.is_last_row_diff = (row == last_row).then(|| is_last_row.eval_packed());
                state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                state.is_transition_eval =
                    Challenge::ExtensionPacking::ONE - state.is_last_row_eval;
                let mut q_bar_eval = q_bar[1];
                let q_bar_diff = q_bar[1] - q_bar[0];
                sum.iter_mut().zip(q).for_each(|(sum, q)| {
                    q_bar_eval += q_bar_diff;
                    state.next_point();
                    let eval = state.eval_packed(air, public_values, alpha_powers);
                    *q = eval - q_bar_eval;
                    *sum += *eq_eval * eval;
                });
                sum
            },
            vec_add,
        )
        .into_iter()
        .map(|eval| eval.ext_sum() * eq_helper.correcting_factor);

    RoundPoly::from_evals(chain![[eval_0, eval_1], extra_evals])
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, fields(dim = %input.height().ilog2()))]
fn compute_round_poly_algo_1<Val, Challenge, A>(
    air: &A,
    degree: usize,
    claim: Challenge,
    public_values: &[Val],
    input: &RowMajorMatrix<Challenge>,
    alpha_powers: &[Challenge],
    is_first_row: &IsFirstRow<Challenge>,
    is_last_row: &IsLastRow<Challenge>,
    eq_helper: &EqHelper<Val, Challenge, Challenge>,
) -> RoundPoly<Challenge>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderWithVal<'t, Val, Challenge, Challenge, Challenge>>,
{
    let width = air.width();
    let last_row = input.height() / 2 - 1;

    let evals = input
        .par_row_chunks(2)
        .zip(eq_helper.evals())
        .enumerate()
        .par_fold_reduce(
            || vec![Challenge::ZERO; degree],
            |mut sum, (row, (main, eq_eval))| {
                let lo = main.row(0);
                let hi = main.row(1);
                let mut state = EvalState::new(width);
                state.main_eval.slice_assign_iter(hi);
                state.main_diff.slice_sub_iter(cloned(&state.main_eval), lo);
                state.is_first_row_eval = Challenge::ZERO;
                state.is_first_row_diff = (row == 0).then(|| -is_first_row.0);
                state.is_last_row_diff = (row == last_row).then_some(is_last_row.0);
                state.is_last_row_eval = state.is_last_row_diff.unwrap_or_default();
                state.is_transition_eval = Challenge::ONE - state.is_last_row_eval;
                sum.iter_mut().enumerate().for_each(|(d, sum)| {
                    if d > 0 {
                        state.next_point();
                    }
                    *sum += *eq_eval * state.eval(air, public_values, alpha_powers)
                });
                sum
            },
            vec_add,
        );

    let evals = evals
        .into_iter()
        .map(|eval| eval * eq_helper.correcting_factor)
        .collect_vec();

    let eval_0 = eq_helper.eval_0(claim, evals[0]);

    RoundPoly(
        vander_mat_inv((0..degree + 1).map(Val::from_usize))
            .rows()
            .map(|row| dot_product(chain![[eval_0], cloned(&evals)], row))
            .collect(),
    )
}
