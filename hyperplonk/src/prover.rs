use alloc::vec::Vec;
use core::ops::Deref;

use itertools::{Itertools, cloned, izip};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    AirMeta, BatchSumcheckProof, EqHelper, FieldSlice, Proof, ProverFolderOnExtension,
    ProverFolderOnExtensionPacking, ProverFolderOnPacking, SymbolicAirBuilder, VerifierInput,
    ZeroSumcheckState,
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
    Val: Field,
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

    // TODO: PCS commit and observe.

    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(input.public_values()));

    // TODO: Prove LogUp with fractional sumchecks.

    let (_z, zero_sumcheck) = prove_zero_sumcheck(
        &metas,
        &inputs,
        traces.iter().map(|mat| mat.as_view()).collect(),
        &mut challenger,
    );

    // TODO: PCS open

    // TODO: Remove the following sanity checks.
    #[cfg(debug_assertions)]
    izip!(&metas, traces, &zero_sumcheck.evals).for_each(|(meta, trace, evals)| {
        let (local, next) = evals.split_at(meta.width);
        let z = &_z[_z.len() - log2_strict_usize(trace.height())..];
        let mut eq_z = crate::eq_poly(z, Challenge::ONE);
        assert_eq!(trace.columnwise_dot_product(&eq_z), local);
        eq_z.rotate_right(1);
        assert_eq!(trace.columnwise_dot_product(&eq_z), next);
    });

    Proof { zero_sumcheck }
}

fn prove_zero_sumcheck<Val, Challenge, A>(
    metas: &[AirMeta],
    inputs: &[VerifierInput<Val, A>],
    traces: Vec<RowMajorMatrixView<Val>>,
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<Challenge>, BatchSumcheckProof<Challenge>)
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    let max_degree = itertools::max(metas.iter().map(|meta| meta.degree)).unwrap_or_default();

    let log_heights = traces
        .iter()
        .map(|mat| log2_strict_usize(mat.height()))
        .collect_vec();
    let max_log_height = itertools::max(cloned(&log_heights)).unwrap();

    log_heights
        .iter()
        .for_each(|log_height| challenger.observe(Val::from_u8(*log_height as u8)));

    let r = (0..max_log_height)
        .map(|_| challenger.sample_algebra_element())
        .collect_vec();
    let eq_helper = EqHelper::new(&r);

    let alpha = challenger.sample_algebra_element::<Challenge>();
    let beta = challenger.sample_algebra_element::<Challenge>();

    let alpha_powers = {
        let max_constraint_count =
            itertools::max(metas.iter().map(|meta| meta.constraint_count)).unwrap();
        let mut alpha_powers = alpha.powers().take(max_constraint_count).collect_vec();
        alpha_powers.reverse();
        alpha_powers
    };

    let mut states = izip!(metas, inputs, traces)
        .map(|(meta, input, trace)| {
            ZeroSumcheckState::new(
                *meta,
                input.air(),
                input.public_values(),
                trace,
                &alpha_powers[alpha_powers.len() - meta.constraint_count..],
                max_log_height,
            )
        })
        .collect_vec();
    let (z, compressed_round_polys) = (0..max_log_height)
        .map(|round| {
            let compressed_round_polys = states
                .iter_mut()
                .map(|state| state.compute_round_poly(round, &eq_helper))
                .collect_vec();

            let compressed_round_poly = compressed_round_polys
                .into_iter()
                .reduce(|acc, mut item| {
                    item.0.resize(max_degree, Challenge::ZERO);
                    item.0[..acc.0.len()].slice_add_assign_scaled_iter(acc.0, beta);
                    item
                })
                .unwrap();

            compressed_round_poly
                .0
                .iter()
                .for_each(|coeff| challenger.observe_algebra_element(*coeff));
            let z_i = challenger.sample_algebra_element();

            states
                .iter_mut()
                .for_each(|state| state.fix_var(round, z_i));

            (z_i, compressed_round_poly)
        })
        .collect();

    let evals = states.into_iter().map(|state| state.into_evals()).collect();

    (
        z,
        BatchSumcheckProof {
            num_vars: log_heights,
            compressed_round_polys,
            evals,
        },
    )
}
