use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::{FieldChallenger, HashChallenger, SerializingChallenger32};
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use p3_hyperplonk::{
    ProverFolderOnExtension, ProverFolderOnExtensionPacking, ProverFolderOnPacking, ProverInput,
    SymbolicAirBuilder, VerifierFolder, prove, verify,
};
use p3_keccak::Keccak256Hash;

type Challenger<Val> = SerializingChallenger32<Val, HashChallenger<u8, Keccak256Hash, 32>>;

#[allow(clippy::multiple_bound_locations)]
pub fn run<
    Val: TwoAdicField + PrimeField32,
    Challenge: ExtensionField<Val>,
    #[cfg(feature = "check-constraints")] A: for<'a> Air<p3_air_ext::DebugConstraintBuilder<'a, Val>>,
    #[cfg(not(feature = "check-constraints"))] A,
>(
    prover_inputs: Vec<ProverInput<Val, A>>,
) where
    A: Clone
        + BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'t> Air<ProverFolderOnPacking<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtension<'t, Val, Challenge>>
        + for<'t> Air<ProverFolderOnExtensionPacking<'t, Val, Challenge>>
        + for<'t> Air<VerifierFolder<'t, Val, Challenge>>,
{
    let verifier_inputs = prover_inputs
        .iter()
        .map(|input| input.to_verifier_input())
        .collect();

    let mut prover_challenger = Challenger::<Val>::from_hasher(Vec::new(), Keccak256Hash {});
    let proof = prove(prover_inputs, &mut prover_challenger);

    let mut verifier_challenger = Challenger::<Val>::from_hasher(Vec::new(), Keccak256Hash {});
    verify::<_, Challenge, _>(verifier_inputs, &proof, &mut verifier_challenger).unwrap();

    assert_eq!(
        prover_challenger.sample_algebra_element::<Challenge>(),
        verifier_challenger.sample_algebra_element::<Challenge>(),
    );
}
