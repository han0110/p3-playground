use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_air_ext::VerifierInput;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_field::extension::BinomialExtensionField;
use p3_hyperplonk::{Trace, keygen, prove_air};
use p3_keccak::Keccak256Hash;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_poseidon2_air::{RoundConstants, generate_trace_rows, num_cols};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove_air/koala_bear_poseidon2");
    group.sample_size(10);

    let mut rng = StdRng::from_os_rng();

    let round_constants = RoundConstants::from_rng(&mut rng);
    let air = &Poseidon2Air(p3_poseidon2_air::Poseidon2Air::new(round_constants.clone()));
    let (_, pk) = keygen([&air]);

    for log_b in 20..21 {
        group.bench_with_input(BenchmarkId::from_parameter(log_b), &log_b, |b, log_b| {
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
            b.iter_batched(
                || Trace::<_, Challenge>::new(trace.clone()),
                |trace| {
                    prove_air(
                        &pk,
                        &[VerifierInput::new(air, Vec::new())],
                        vec![trace],
                        &[],
                        &[],
                        &[Default::default()],
                        Challenger::from_hasher(Vec::new(), Keccak256Hash {}),
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
