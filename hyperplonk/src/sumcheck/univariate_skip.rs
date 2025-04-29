use alloc::vec;
use alloc::vec::Vec;

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
use p3_util::log2_ceil_usize;
use tracing::{info_span, instrument};

use crate::{
    AirMeta, AirTrace, EvalSumcheckProver, FieldSlice, IsFirstRow, IsLastRow, IsTransition,
    PackedExtensionValue, ProverFolderGeneric, ProverFolderOnExtension,
    ProverFolderOnExtensionPacking, ProverFolderOnPacking, RegularSumcheckProver, RoundPoly,
    eq_poly_packed, vec_add,
};

pub(crate) struct UnivariateSkipProver<'a, Val: Field, Challenge: ExtensionField<Val>, A> {
    meta: AirMeta,
    air: &'a A,
    public_values: &'a [Val],
    alpha_powers: &'a [Challenge],
    trace: AirTrace<Val, Challenge>,
    skip_rounds: usize,
    round_poly: RoundPoly<Challenge>,
}

impl<'a, Val, Challenge, A> UnivariateSkipProver<'a, Val, Challenge, A>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>,
{
    pub(crate) fn new(
        meta: AirMeta,
        air: &'a A,
        public_values: &'a [Val],
        alpha_powers: &'a [Challenge],
        trace: RowMajorMatrixView<Val>,
        skip_rounds: usize,
    ) -> Self {
        assert!(trace.height() >= Val::Packing::WIDTH << skip_rounds);
        let trace = AirTrace::new(trace);
        Self {
            meta,
            air,
            public_values,
            alpha_powers,
            trace,
            skip_rounds,
            round_poly: Default::default(),
        }
    }

    #[instrument(skip_all, name = "compute univariate skip round poly", fields(log_h = %self.trace.log_height()))]
    pub(crate) fn compute_round_poly(&mut self, r: &[Challenge]) -> RoundPoly<Challenge> {
        let AirTrace::Packing(trace) = &self.trace else {
            unimplemented!()
        };

        let round_poly = {
            let quotient_degree = self.meta.univariate_degree.saturating_sub(1);
            let added_bits = log2_ceil_usize(quotient_degree);
            let sels = domain(self.skip_rounds)
                .selectors_on_coset(quotient_domain(self.skip_rounds, added_bits));
            let eq_r_packed = eq_poly_packed(r);
            let quotient_values = trace
                .par_row_chunks(1 << self.skip_rounds)
                .zip(&eq_r_packed)
                .enumerate()
                .par_fold_reduce(
                    || {
                        vec![
                            Challenge::ExtensionPacking::ZERO;
                            1 << (self.skip_rounds + added_bits)
                        ]
                    },
                    |mut quotient_values, (chunk, (trace, scalar))| {
                        quotient_values.slice_add_assign_scaled_iter(
                            compute_quotient_values(
                                self.meta,
                                self.air,
                                self.public_values,
                                trace,
                                self.alpha_powers,
                                &sels,
                                added_bits,
                                chunk == 0,
                                chunk == eq_r_packed.len() - 1,
                            ),
                            *scalar,
                        );
                        quotient_values
                    },
                    vec_add,
                )
                .into_par_iter()
                .map(|v| v.ext_sum())
                .collect::<Vec<_>>();
            let quotient_coeffs = Radix2DitParallel::default().coset_idft_batch(
                RowMajorMatrix::new_col(quotient_values).flatten_to_base(),
                Val::GENERATOR,
            );
            RoundPoly(
                quotient_coeffs
                    .values
                    .par_chunks(Challenge::DIMENSION)
                    .map(|chunk| Challenge::from_basis_coefficients_slice(chunk).unwrap())
                    .take(quotient_degree << self.skip_rounds)
                    .collect(),
            )
        };

        self.round_poly = round_poly.clone();
        round_poly
    }

    pub(crate) fn to_regular_prover(
        &self,
        x: Challenge,
        max_regular_rounds: usize,
    ) -> RegularSumcheckProver<'a, Val, Challenge, A>
    where
        A: for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
            + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
            + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
    {
        let claim = self.round_poly.subclaim(x) * (x.exp_power_of_2(self.skip_rounds) - Val::ONE);
        let regular_rounds = self.trace.log_height() - self.skip_rounds;
        let trace = info_span!("fix univariate skip var").in_scope(|| {
            let lagrange_evals = lagrange_evals(self.skip_rounds, x);
            self.trace.fix_lo_scalars(&lagrange_evals)
        });
        let sels = selectors_at_point(self.skip_rounds, x);
        RegularSumcheckProver {
            air: self.air,
            meta: self.meta,
            public_values: self.public_values,
            alpha_powers: self.alpha_powers,
            starting_round: max_regular_rounds - regular_rounds,
            claim,
            trace,
            is_first_row: IsFirstRow(sels.is_first_row),
            is_last_row: IsLastRow(sels.is_last_row),
            is_transition: IsTransition(sels.is_transition),
            round_poly: Default::default(),
        }
    }

    pub(crate) fn into_univariate_eval_prover<'b>(
        self,
        x: Challenge,
        z: &[Challenge],
        evals: &[Challenge],
        gamma_powers: &'b [Challenge],
        max_skip_rounds: usize,
    ) -> EvalSumcheckProver<'b, Challenge> {
        let claim = dot_product(cloned(evals), cloned(gamma_powers));
        let trace = info_span!("fix high vars").in_scope(|| match self.trace.fix_hi_vars(z) {
            AirTrace::Extension(trace) => trace,
            _ => unimplemented!(),
        });
        let weight = lagrange_evals(self.skip_rounds, x);
        EvalSumcheckProver {
            trace,
            weight,
            gamma_powers,
            claim,
            starting_round: max_skip_rounds - self.skip_rounds,
            round_poly: Default::default(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_quotient_values<Val, Challenge, A>(
    meta: AirMeta,
    air: &A,
    public_values: &[Val],
    trace: RowMajorMatrixView<Val::Packing>,
    alpha_powers: &[Challenge],
    sels: &LagrangeSelectors<Vec<Val>>,
    added_bits: usize,
    is_first_chunk: bool,
    is_last_chunk: bool,
) -> Vec<Challenge::ExtensionPacking>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>,
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
    (0..trace_lde.height())
        .into_par_iter()
        .map(|row| {
            let row_slice = trace_lde.row_slice(row).unwrap();
            let main = RowMajorMatrixView::new(Val::Packing::pack_slice(&row_slice), meta.width);
            let sels = selectors_at_row(sels, is_first_chunk, is_last_chunk, row);
            let mut folder = ProverFolderGeneric {
                main,
                public_values,
                is_first_row: sels.is_first_row,
                is_last_row: sels.is_last_row,
                is_transition: sels.is_transition,
                alpha_powers,
                accumulator: Challenge::ExtensionPacking::ZERO,
                constraint_index: 0,
            };
            air.eval(&mut folder);
            folder.accumulator * sels.inv_vanishing
        })
        .collect()
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
    log_height: usize,
    z: Challenge,
) -> Vec<Challenge> {
    let subgroup = cyclic_subgroup_coset_known_order(
        Val::two_adic_generator(log_height),
        Val::ONE,
        1 << log_height,
    )
    .collect_vec();
    let vanishing_over_height = (z.exp_power_of_2(log_height) - Challenge::ONE)
        * Val::ONE.halve().exp_u64(log_height as u64);
    let diff_invs =
        batch_multiplicative_inverse(&subgroup.par_iter().map(|&x| z - x).collect::<Vec<_>>());
    subgroup
        .into_par_iter()
        .zip(diff_invs)
        .map(|(x, diff_inv)| vanishing_over_height * diff_inv * x)
        .collect()
}
