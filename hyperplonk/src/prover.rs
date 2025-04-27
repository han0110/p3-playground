use alloc::vec::Vec;
use core::ops::Deref;

use itertools::{Itertools, chain, cloned, izip};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    AirMeta, BatchSumcheckProof, EqHelper, FieldSlice, Proof, ProverFolderOnExtension,
    ProverFolderOnExtensionPacking, ProverFolderOnPacking, RegularSumcheckProver,
    SymbolicAirBuilder, UnivariateSkipProof, UnivariateSkipProver, VerifierInput, ZeroCheckProof,
};

#[derive(Clone, Debug)]
pub struct ProverInput<Val, A> {
    pub(crate) inner: VerifierInput<Val, A>,
    pub(crate) trace: RowMajorMatrix<Val>,
}

impl<Val, A> Deref for ProverInput<Val, A> {
    type Target = VerifierInput<Val, A>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Val: Field, A> ProverInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>, trace: RowMajorMatrix<Val>) -> Self
    where
        A: BaseAirWithPublicValues<Val>,
    {
        assert_eq!(air.width(), trace.width());
        Self {
            inner: VerifierInput::new(air, public_values),
            trace,
        }
    }
}

#[instrument(skip_all)]
pub fn prove<Val, Challenge, A>(
    inputs: Vec<ProverInput<Val, A>>,
    mut challenger: impl FieldChallenger<Val>,
) -> Proof<Challenge>
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    assert!(!inputs.is_empty());

    // TODO: Preprocess the meta.
    let metas = inputs
        .iter()
        .map(|input| AirMeta::new(input.air()))
        .collect_vec();

    let (inputs, traces) = inputs
        .into_iter()
        .map(|input| (input.inner, input.trace))
        .collect::<(Vec<_>, Vec<_>)>();
    let log_heights = traces
        .iter()
        .map(|mat| log2_strict_usize(mat.height()))
        .collect_vec();

    // TODO: PCS commit and observe.

    cloned(&log_heights).for_each(|log_height| challenger.observe(Val::from_u8(log_height as u8)));
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    // TODO: Prove LogUp with fractional-sum check.

    let (_zs, zero_check) = prove_zero_check(
        &metas,
        &inputs,
        traces.iter().map(|mat| mat.as_view()).collect(),
        &log_heights,
        &mut challenger,
    );

    // TODO: PCS open

    // TODO: Remove the following sanity checks.
    #[cfg(debug_assertions)]
    izip!(&metas, &traces, &zero_check.univariate_skips, &_zs)
        .enumerate()
        .for_each(|(idx, (meta, trace, univariate_skip, z))| {
            let evals = if univariate_skip.skip_rounds > 0 {
                &zero_check.univariate_eval_sumcheck.evals[idx]
            } else {
                &zero_check.regular_sumcheck.evals[idx]
            };
            let (local, next) = evals.split_at(meta.width);
            let mut eq_z = crate::eq_poly(z, Challenge::ONE);
            assert_eq!(trace.columnwise_dot_product(&eq_z), local);
            eq_z.rotate_right(1);
            assert_eq!(trace.columnwise_dot_product(&eq_z), next);
        });

    Proof {
        log_heights,
        zero_check,
    }
}

fn prove_zero_check<Val, Challenge, A>(
    metas: &[AirMeta],
    inputs: &[VerifierInput<Val, A>],
    traces: Vec<RowMajorMatrixView<Val>>,
    log_heights: &[usize],
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<Vec<Challenge>>, ZeroCheckProof<Challenge>)
where
    Val: TwoAdicField + Ord,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    let log_packing_width = log2_strict_usize(Val::Packing::WIDTH);

    let skip_rounds = cloned(log_heights)
        .map(|log_height| {
            // TODO: Find a better way to choose the optimal rounds to skip automatically.
            const SKIP_ROUNDS: usize = 6;
            (log_height >= SKIP_ROUNDS + log_packing_width)
                .then_some(SKIP_ROUNDS)
                .unwrap_or_default()
        })
        .collect_vec();
    let regular_rounds = izip!(cloned(log_heights), cloned(&skip_rounds))
        .map(|(log_height, skip_rounds)| log_height - skip_rounds)
        .collect_vec();
    let max_skip_rounds = itertools::max(cloned(&skip_rounds)).unwrap();
    let max_regular_rounds = itertools::max(cloned(&regular_rounds)).unwrap();

    let r = (0..max_regular_rounds)
        .map(|_| challenger.sample_algebra_element())
        .collect_vec();

    let alpha = challenger.sample_algebra_element::<Challenge>();

    let alpha_powers = {
        let max_constraint_count =
            itertools::max(metas.iter().map(|meta| meta.constraint_count)).unwrap();
        let mut alpha_powers = alpha.powers().take(max_constraint_count).collect_vec();
        alpha_powers.reverse();
        alpha_powers
    };

    let mut univariate_skip_provers = izip!(cloned(metas), inputs, &traces, cloned(&skip_rounds))
        .map(|(meta, input, trace, skip_rounds)| {
            (skip_rounds > 0).then(|| {
                UnivariateSkipProver::new(
                    meta,
                    input.air(),
                    input.public_values(),
                    &alpha_powers[alpha_powers.len() - meta.constraint_count..],
                    trace.as_view(),
                    skip_rounds,
                )
            })
        })
        .collect_vec();

    let univariate_skips = izip!(
        univariate_skip_provers.iter_mut(),
        cloned(&skip_rounds),
        cloned(&regular_rounds)
    )
    .map(|(prover, skip_rounds, regular_rounds)| {
        prover
            .as_mut()
            .map(|prover| {
                let round_poly = prover.compute_round_poly(&r[r.len() - regular_rounds..]);
                UnivariateSkipProof {
                    skip_rounds,
                    round_poly,
                }
            })
            .unwrap_or_default()
    })
    .collect_vec();

    univariate_skips.iter().for_each(|univariate_skip| {
        challenger.observe(Val::from_u8(univariate_skip.skip_rounds as u8));
        cloned(&univariate_skip.round_poly.0)
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
    });

    let x = challenger.sample_algebra_element();

    let mut regular_provers = izip!(cloned(metas), inputs, traces, &univariate_skip_provers)
        .map(|(meta, input, trace, univariate_skip_prover)| {
            if let Some(univariate_skip_prover) = univariate_skip_prover {
                univariate_skip_prover.to_regular_prover(x, max_regular_rounds)
            } else {
                RegularSumcheckProver::new(
                    meta,
                    input.air(),
                    input.public_values(),
                    Challenge::ZERO,
                    trace,
                    &alpha_powers[alpha_powers.len() - meta.constraint_count..],
                    max_regular_rounds,
                )
            }
        })
        .collect_vec();

    let beta = challenger.sample_algebra_element::<Challenge>();
    let beta_powers = beta.powers().take(metas.len()).collect_vec();

    let (z, regular_sumcheck) = {
        let max_multivariate_degree =
            itertools::max(metas.iter().map(|meta| meta.multivariate_degree)).unwrap_or_default();
        let eq_helper = EqHelper::new(&r);

        let (z, compressed_round_polys) = (0..max_regular_rounds)
            .map(|round| {
                let compressed_round_polys = regular_provers
                    .iter_mut()
                    .map(|state| state.compute_eq_weighted_round_poly(round, &eq_helper))
                    .collect_vec();

                let compressed_round_poly = izip!(compressed_round_polys, cloned(&beta_powers))
                    .map(|(mut compressed_round_poly, scalar)| {
                        compressed_round_poly.0.slice_scale(scalar);
                        compressed_round_poly
                    })
                    .reduce(|mut acc, item| {
                        acc.0.resize(max_multivariate_degree, Challenge::ZERO);
                        acc.0.slice_add_assign(&item.0);
                        acc
                    })
                    .unwrap();

                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_i = challenger.sample_algebra_element();

                regular_provers
                    .iter_mut()
                    .for_each(|state| state.fix_var(round, z_i));

                (z_i, compressed_round_poly)
            })
            .collect::<(Vec<_>, Vec<_>)>();

        let evals = regular_provers
            .into_iter()
            .map(|state| state.into_evals())
            .collect_vec();

        cloned(evals.iter().flatten()).for_each(|eval| challenger.observe_algebra_element(eval));

        (
            z,
            BatchSumcheckProof {
                compressed_round_polys,
                evals,
            },
        )
    };

    let gamma: Challenge = challenger.sample_algebra_element();
    let theta: Challenge = challenger.sample_algebra_element();

    let gamma_powers = {
        let max_width = itertools::max(metas.iter().map(|meta| 2 * meta.width)).unwrap();
        let mut gamma_powers = gamma.powers().take(max_width).collect_vec();
        gamma_powers.reverse();
        gamma_powers
    };
    let theta_powers = theta.powers().take(metas.len()).collect_vec();

    let mut eval_provers = izip!(
        metas,
        univariate_skip_provers,
        &regular_sumcheck.evals,
        cloned(&regular_rounds)
    )
    .map(|(meta, prover, evals, regular_rounds)| {
        prover.map(|prover| {
            prover.into_univariate_eval_prover(
                x,
                &z[z.len() - regular_rounds..],
                evals,
                &gamma_powers[gamma_powers.len() - 2 * meta.width..],
                max_skip_rounds,
            )
        })
    })
    .collect_vec();

    let (z_prime, univariate_eval_sumcheck) = {
        let (z_pirme, compressed_round_polys) = (0..max_skip_rounds)
            .map(|round| {
                let compressed_round_polys = eval_provers
                    .iter_mut()
                    .map(|prover| {
                        prover
                            .as_mut()
                            .map(|prover| prover.compute_round_poly(round))
                            .unwrap_or_default()
                    })
                    .collect_vec();

                let compressed_round_poly = izip!(compressed_round_polys, cloned(&theta_powers))
                    .map(|(mut compressed_round_poly, scalar)| {
                        compressed_round_poly.0.slice_scale(scalar);
                        compressed_round_poly
                    })
                    .reduce(|mut acc, item| {
                        acc.0.resize(2, Challenge::ZERO);
                        acc.0.slice_add_assign(&item.0);
                        acc
                    })
                    .unwrap();

                cloned(&compressed_round_poly.0)
                    .for_each(|coeff| challenger.observe_algebra_element(coeff));
                let z_prime_i = challenger.sample_algebra_element();

                eval_provers
                    .iter_mut()
                    .flatten()
                    .for_each(|prover| prover.fix_var(round, z_prime_i));

                (z_prime_i, compressed_round_poly)
            })
            .collect::<(Vec<_>, Vec<_>)>();
        let evals = eval_provers
            .into_iter()
            .map(|prover| prover.map(|prover| prover.into_evals()).unwrap_or_default())
            .collect_vec();

        cloned(evals.iter().flatten()).for_each(|eval| challenger.observe_algebra_element(eval));

        (
            z_pirme,
            BatchSumcheckProof {
                compressed_round_polys,
                evals,
            },
        )
    };

    let zs = izip!(skip_rounds, regular_rounds)
        .map(|(skip_rounds, regular_rounds)| {
            chain![
                &z_prime[z_prime.len() - skip_rounds..],
                &z[z.len() - regular_rounds..]
            ]
            .copied()
            .collect()
        })
        .collect();

    (
        zs,
        ZeroCheckProof {
            univariate_skips,
            regular_sumcheck,
            univariate_eval_sumcheck,
        },
    )
}
