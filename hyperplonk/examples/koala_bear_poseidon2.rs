use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_field::extension::BinomialExtensionField;
use p3_hyperplonk::{ProverInput, VerifierInput, keygen, prove, verify};
use p3_keccak::Keccak256Hash;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_poseidon2_air::{RoundConstants, generate_trace_rows, num_cols};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracing_forest::ForestLayer;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, Registry};

#[cfg_attr(target_family = "unix", global_allocator)]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

type Val = KoalaBear;
type Challenge = BinomialExtensionField<Val, 4>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, Keccak256Hash, 32>>;

const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

pub struct Poseidon2Air(
    p3_poseidon2_air::Poseidon2Air<
        Val,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
);

impl<F> BaseAir<F> for &Poseidon2Air {
    fn width(&self) -> usize {
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
    }
}

impl<F> BaseAirWithPublicValues<F> for &Poseidon2Air {}

impl<AB: AirBuilder<F = Val>> Air<AB> for &Poseidon2Air {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        self.0.eval(builder);
    }
}

fn main() {
    let mut rng = StdRng::from_os_rng();
    let round_constants = RoundConstants::from_rng(&mut rng);
    let air = &Poseidon2Air(p3_poseidon2_air::Poseidon2Air::new(round_constants.clone()));
    let (vk, pk) = keygen([&air]);

    let log_b = 20;
    let trace = generate_trace_rows::<
        _,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(
        (0..1 << log_b).map(|_| rng.random()).collect(),
        &round_constants,
        0,
    );

    let start = Instant::now();
    while Instant::now().duration_since(start).as_secs() < 3 {
        let prover_inputs = vec![ProverInput::new(air, Vec::new(), trace.clone())];
        let challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
        prove::<_, Challenge, _>(&pk, prover_inputs, challenger);
    }

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,p3_dft=warn"));

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let prover_inputs = vec![ProverInput::new(air, Vec::new(), trace)];
    let challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
    let proof = prove::<_, Challenge, _>(&pk, prover_inputs, challenger);

    let verifier_inputs = vec![VerifierInput::new(air, Vec::new())];
    let challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
    verify(&vk, verifier_inputs, &proof, challenger).unwrap();
}
