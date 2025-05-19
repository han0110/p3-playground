use itertools::Itertools;
use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{TwoAdicFriPcs, create_test_fri_config};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark_ext::{
    InteractionBuilder, ProverInput, StarkConfig, VerifierInput, keygen, prove, verify,
};
use rand::{Rng, rng};

struct SendingAir;

impl<F> BaseAir<F> for SendingAir {
    fn width(&self) -> usize {
        1 // [value]
    }
}

impl<AB: InteractionBuilder> Air<AB> for SendingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        if !AB::ONLY_INTERACTION {
            builder.assert_eq(local[0].into().square(), local[0].into().square());
        }
        builder.push_send(0, [local[0]], AB::Expr::ONE);
    }
}

struct ReceivingAir;

impl<F> BaseAir<F> for ReceivingAir {
    fn width(&self) -> usize {
        2 // [value, mult]
    }
}

impl<AB: InteractionBuilder> Air<AB> for ReceivingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        if !AB::ONLY_INTERACTION {
            builder.assert_eq(local[0].into().square(), local[0].into().square());
        }
        builder.push_receive(0, [local[0]], local[1]);
    }
}

enum MyAir {
    Sending(SendingAir),
    Receiving(ReceivingAir),
}

impl<F> BaseAir<F> for MyAir {
    fn width(&self) -> usize {
        match self {
            Self::Sending(inner) => BaseAir::<F>::width(inner),
            Self::Receiving(inner) => BaseAir::<F>::width(inner),
        }
    }
}

impl<F> BaseAirWithPublicValues<F> for MyAir {
    fn num_public_values(&self) -> usize {
        0
    }
}

impl<AB: InteractionBuilder> Air<AB> for MyAir {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Sending(inner) => inner.eval(builder),
            Self::Receiving(inner) => inner.eval(builder),
        }
    }
}

fn generate_sending_trace<F: Field>(n: usize, mut rng: impl Rng) -> RowMajorMatrix<F> {
    let mut trace = RowMajorMatrix::new_col(F::zero_vec(n));
    trace
        .values
        .iter_mut()
        .for_each(|cell| *cell = F::from_u8(rng.random()));
    trace
}

fn generate_receiving_trace<F: Field>(sending_trace: &RowMajorMatrix<F>) -> RowMajorMatrix<F> {
    let counts = sending_trace.values.iter().counts();
    let mut trace = RowMajorMatrix::new(F::zero_vec(2 * counts.len().next_power_of_two()), 2);
    trace
        .rows_mut()
        .zip(counts)
        .for_each(|(row, (value, mult))| {
            row[0] = *value;
            row[1] = F::from_usize(mult);
        });
    trace
}

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn do_test(sending_trace: RowMajorMatrix<Val>, receiving_trace: RowMajorMatrix<Val>) {
    let perm = Perm::new_from_rng_128(&mut rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_config = create_test_fri_config(challenge_mmcs, 0);
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);

    let (vk, pk) = keygen::<Val, _>(
        3,
        &[MyAir::Sending(SendingAir), MyAir::Receiving(ReceivingAir)],
    );

    let proof = prove(
        &config,
        &pk,
        vec![
            ProverInput::new(MyAir::Sending(SendingAir), Vec::new(), sending_trace),
            ProverInput::new(MyAir::Receiving(ReceivingAir), Vec::new(), receiving_trace),
        ],
    );
    verify(
        &config,
        &vk,
        vec![
            VerifierInput::new(MyAir::Sending(SendingAir), Vec::new()),
            VerifierInput::new(MyAir::Receiving(ReceivingAir), Vec::new()),
        ],
        &proof,
    )
    .expect("verification failed");
}

#[test]
fn test_valid() {
    let sending_trace = generate_sending_trace::<Val>(1 << 10, rng());
    let receiving_trace = generate_receiving_trace::<Val>(&sending_trace);
    do_test(sending_trace, receiving_trace);
}

#[test]
#[cfg(feature = "check-constraints")]
#[should_panic(
    expected = "Interaction multiset equality check failed.\nBus 0 failed to balance the multiplicities for fields: [256]. The bus records for this were:\n  Air index: 1, interaction type: Receive, count: 1"
)]
fn test_invalid() {
    let sending_trace = generate_sending_trace::<Val>(1 << 10, rng());
    let mut receiving_trace = generate_receiving_trace::<Val>(&sending_trace);
    let last_row = receiving_trace.rows_mut().last().unwrap();
    last_row[0] = Val::from_i32(256);
    last_row[1] = Val::ONE;
    do_test(sending_trace, receiving_trace);
}
