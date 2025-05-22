use core::iter::repeat_with;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_air_ext::{InteractionBuilder, VerifierInput};
use p3_challenger::{FieldChallenger, HashChallenger, SerializingChallenger32};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_hyperplonk::{Trace, keygen, prove_fractional_sum};
use p3_keccak::Keccak256Hash;
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg_attr(target_family = "unix", global_allocator)]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

type Val = KoalaBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, Keccak256Hash, 32>>;

#[derive(Clone, Copy)]
pub struct InteractionAir;

const INTERACTION_COUNT: usize = 20;

impl<F> BaseAir<F> for InteractionAir {
    fn width(&self) -> usize {
        1
    }
}

impl<F> BaseAirWithPublicValues<F> for InteractionAir {}

impl<AB: InteractionBuilder> Air<AB> for InteractionAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        for _ in 0..INTERACTION_COUNT / 2 {
            builder.push_send(0, [local[0]], AB::Expr::ONE);
            builder.push_receive(0, [local[0]], AB::Expr::ONE);
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("fractional_sum/n={INTERACTION_COUNT}"));
    group.sample_size(10);

    let mut rng = StdRng::from_os_rng();

    let air = InteractionAir;
    let (_, pk) = keygen::<Val, _>([&air]);

    for log_b in 20..21 {
        group.bench_with_input(BenchmarkId::from_parameter(log_b), &log_b, |b, log_b| {
            let trace =
                RowMajorMatrix::new(repeat_with(|| rng.random()).take(1 << log_b).collect(), 1);
            b.iter_batched(
                || Trace::<_, Challenge>::new(trace.clone()),
                |trace| {
                    let mut challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
                    let gamma: Challenge = challenger.sample_algebra_element();
                    prove_fractional_sum(
                        &pk,
                        &[VerifierInput::new(air, Vec::new())],
                        &[trace],
                        &[],
                        &[gamma],
                        challenger,
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
