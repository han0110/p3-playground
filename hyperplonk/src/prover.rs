use alloc::vec::Vec;

use itertools::{Itertools, chain};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    AirMeta, EqHelper, Proof, ProverFolderOnExtension, ProverFolderOnExtensionPacking,
    ProverFolderOnPacking, SumcheckProof, SymbolicAirBuilder, ZeroSumcheckState,
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
        + for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    assert_eq!(air.width(), input.width());

    // TODO: Preprocess the meta.
    let meta = AirMeta::new(air);

    // TODO: PCS commit and observe.

    challenger.observe_slice(public_values);

    // TODO: Prove LogUp with fractional sumchecks.

    let (_z, zero_sumcheck) =
        prove_zero_sumcheck(air, meta, public_values, input.as_view(), &mut challenger);

    // TODO: PCS open

    // TODO: Remove the following sanity checks.
    #[cfg(debug_assertions)]
    {
        let (local, next) = zero_sumcheck.evals.split_at(air.width());
        let mut eq_z = crate::eq_poly(&_z, Challenge::ONE);
        assert_eq!(input.columnwise_dot_product(&eq_z), local);
        eq_z.rotate_right(1);
        assert_eq!(input.columnwise_dot_product(&eq_z), next);
    }

    Proof { zero_sumcheck }
}

pub fn prove_zero_sumcheck<Val, Challenge, A>(
    air: &A,
    meta: AirMeta,
    public_values: &[Val],
    input: RowMajorMatrixView<Val>,
    mut challenger: impl FieldChallenger<Val>,
) -> (Vec<Challenge>, SumcheckProof<Challenge>)
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    A: for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>,
{
    let log_height = log2_strict_usize(input.height());
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
    let eq_helper = EqHelper::new(&r);

    let max_constraint_count = meta.constraint_count;
    let alpha = challenger.sample_algebra_element::<Challenge>();
    let mut alpha_powers = alpha.powers().take(max_constraint_count).collect_vec();
    alpha_powers.reverse();

    let mut z = Vec::with_capacity(log_height);
    let mut compressed_round_polys = Vec::with_capacity(log_height);

    let max_log_height = log_height;
    let mut state = ZeroSumcheckState::new(
        air,
        meta,
        public_values,
        &alpha_powers,
        max_log_height,
        Challenge::ZERO,
        input,
    );

    for round in 0..max_log_height {
        let round_poly = state.compute_round_poly(round, &eq_helper);

        round_poly
            .iter_compressed()
            .for_each(|coeff| challenger.observe_algebra_element(coeff));
        let z_i = challenger.sample_algebra_element();

        state.fix_var(round, z_i);

        compressed_round_polys.push(round_poly.into_compressed());
        z.push(z_i);
    }

    let evals = state.into_evals();

    (
        z,
        SumcheckProof {
            num_vars: log_height,
            compressed_round_polys,
            evals,
        },
    )
}
