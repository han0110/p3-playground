use itertools::Itertools;
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use p3_hyperplonk::{
    HyperPlonkConfig, ProverConstraintFolderOnExtension, ProverConstraintFolderOnExtensionPacking,
    ProverConstraintFolderOnPacking, ProverInput, ProverInteractionFolderOnExtension,
    ProverInteractionFolderOnPacking, SymbolicAirBuilder, VerifierConstraintFolder, keygen, prove,
    verify,
};
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirPcs};

type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type Dft<Val> = Radix2DitParallel<Val>;
type Pcs<Val, Dft> = WhirPcs<Val, Dft, FieldHash, MyCompress, 32>;
type Challenger<Val> = SerializingChallenger32<Val, HashChallenger<u8, Keccak256Hash, 32>>;

#[allow(clippy::multiple_bound_locations)]
pub fn run<
    Val,
    Challenge,
    #[cfg(feature = "check-constraints")] A: for<'a> Air<p3_air_ext::DebugConstraintBuilder<'a, Val>>,
    #[cfg(not(feature = "check-constraints"))] A,
>(
    prover_inputs: Vec<ProverInput<Val, A>>,
) where
    Val: TwoAdicField + PrimeField32,
    Challenge: TwoAdicField + ExtensionField<Val>,
    A: Clone
        + BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<ProverInteractionFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverInteractionFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverConstraintFolderOnExtensionPacking<'t, Val, Challenge>>
        + for<'t> Air<VerifierConstraintFolder<'t, Val, Challenge>>,
{
    let config = {
        let dft = Dft::default();
        let security_level = 60;
        let pow_bits = 0;
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::Constant(4),
            merkle_hash: field_hash,
            merkle_compress: compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        HyperPlonkConfig::new(
            Pcs::new(dft, whir_params),
            Challenger::from_hasher(Vec::new(), Keccak256Hash {}),
        )
    };

    let verifier_inputs = prover_inputs
        .iter()
        .map(|input| input.to_verifier_input())
        .collect_vec();

    let (vk, pk) = keygen(verifier_inputs.iter().map(|input| input.air()));

    let proof = prove(&config, &pk, prover_inputs);

    verify(&config, &vk, verifier_inputs, &proof).unwrap();
}
