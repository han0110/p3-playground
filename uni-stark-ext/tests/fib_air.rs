use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{TwoAdicFriPcs, create_test_fri_config};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark_ext::{ProverInput, StarkConfig, VerifierInput, keygen, prove, verify};
use rand::rng;

/// For testing the public values feature
pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<F> BaseAirWithPublicValues<F> for FibonacciAir {
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (main.row_slice(0).unwrap(), main.row_slice(1).unwrap());
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left, a);
        when_first_row.assert_eq(local.right, b);

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.right, next.left);

        // b' <- a + b
        when_transition.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * NUM_FIBONACCI_COLS), NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibonacciRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibonacciRow::new(F::from_u64(a), F::from_u64(b));

    for i in 1..n {
        rows[i].left = rows[i - 1].right;
        rows[i].right = rows[i - 1].left + rows[i - 1].right;
    }

    trace
}

const NUM_FIBONACCI_COLS: usize = 2;

pub struct FibonacciRow<F> {
    pub left: F,
    pub right: F,
}

impl<F> FibonacciRow<F> {
    const fn new(left: F, right: F) -> Self {
        Self { left, right }
    }
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
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

/// n-th Fibonacci number expected to be x
fn test_public_value_impl(n: usize, x: u64, log_final_poly_len: usize) {
    let perm = Perm::new_from_rng_128(&mut rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_config = create_test_fri_config(challenge_mmcs, log_final_poly_len);
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);
    let (vk, pk) = keygen::<Val, _>(3, &[FibonacciAir {}]);
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];
    let proof = prove(
        &config,
        &pk,
        vec![ProverInput::new(FibonacciAir {}, pis.clone(), trace)],
    );
    verify(
        &config,
        &vk,
        vec![VerifierInput::new(FibonacciAir {}, pis)],
        &proof,
    )
    .expect("verification failed");
}

#[test]
fn test_one_row_trace() {
    test_public_value_impl(1, 1, 0);
}

#[test]
fn test_public_value() {
    test_public_value_impl(1 << 3, 21, 2);
}

#[test]
#[cfg(feature = "check-constraints")]
#[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
fn test_incorrect_public_value() {
    let perm = Perm::new_from_rng_128(&mut rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_config = create_test_fri_config(challenge_mmcs, 1);
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);
    let (_, pk) = keygen::<Val, _>(3, &[FibonacciAir {}]);
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let pis = vec![
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::from_u32(123_123), // incorrect result
    ];
    prove(
        &config,
        &pk,
        vec![ProverInput::new(FibonacciAir {}, pis.clone(), trace)],
    );
}
