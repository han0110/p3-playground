use core::iter::repeat_with;

use itertools::{Itertools, chain};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues};
use p3_challenger::{FieldChallenger, HashChallenger, SerializingChallenger32};
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_hyperplonk::{
    ProverFolderOnExtension, ProverFolderOnExtensionPacking, ProverFolderOnPacking, ProverInput,
    SymbolicAirBuilder, VerifierFolder, prove, verify,
};
use p3_keccak::Keccak256Hash;
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

type Val = KoalaBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, Keccak256Hash, 32>>;

#[derive(Clone, Copy)]
enum MyAir {
    GrandSum { width: usize },
    GrandProduct { width: usize },
}

impl<F> BaseAir<F> for MyAir {
    fn width(&self) -> usize {
        match self {
            Self::GrandSum { width } => *width,
            Self::GrandProduct { width } => *width,
        }
    }
}

impl<F> BaseAirWithPublicValues<F> for MyAir {
    fn num_public_values(&self) -> usize {
        1
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for MyAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let grand_output = builder.public_values()[0];
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        let output = match self {
            Self::GrandSum { .. } => local.iter().copied().map_into().sum::<AB::Expr>(),
            Self::GrandProduct { .. } => local.iter().copied().map_into().product::<AB::Expr>(),
        };

        builder.when_transition().assert_eq(output.clone(), next[0]);
        builder
            .when_last_row()
            .assert_eq(output.clone(), grand_output);
    }
}

impl MyAir {
    fn generate_trace_rows<F: Field>(
        &self,
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (RowMajorMatrix<F>, F)
    where
        StandardUniform: Distribution<F>,
    {
        let width = BaseAir::<F>::width(self);
        let mut output = None;
        let input = RowMajorMatrix::new(
            (0..1 << num_vars)
                .flat_map(|_| {
                    let row = chain![output, repeat_with(|| rng.random())]
                        .take(width)
                        .collect_vec();
                    output = Some(match self {
                        Self::GrandSum { .. } => row.iter().copied().sum(),
                        Self::GrandProduct { .. } => row.iter().copied().product(),
                    });
                    row
                })
                .collect(),
            width,
        );
        (input, output.unwrap())
    }
}

fn run<A>(prover_inputs: Vec<ProverInput<Val, A>>)
where
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
        .map(|input| (**input).clone())
        .collect();

    let mut prover_challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
    let proof = prove(prover_inputs, &mut prover_challenger);

    let mut verifier_challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
    verify::<_, Challenge, _>(verifier_inputs, &proof, &mut verifier_challenger).unwrap();

    assert_eq!(
        prover_challenger.sample_algebra_element::<Challenge>(),
        verifier_challenger.sample_algebra_element::<Challenge>(),
    );
}

#[test]
fn single_sum() {
    let mut rng = StdRng::from_os_rng();
    for (num_vars, width) in (0..12).cartesian_product(2..5) {
        let air = MyAir::GrandSum { width };
        let (trace, output) = air.generate_trace_rows(num_vars, &mut rng);
        let public_values = vec![output];
        run(vec![ProverInput::new(air, public_values, trace)]);
    }
}

#[test]
fn single_product() {
    let mut rng = StdRng::from_os_rng();
    for (num_vars, width) in (0..12).cartesian_product(2..5) {
        let air = MyAir::GrandProduct { width };
        let (trace, output) = air.generate_trace_rows(num_vars, &mut rng);
        let public_values = vec![output];
        run(vec![ProverInput::new(air, public_values, trace)]);
    }
}

#[test]
fn multiple_mixed() {
    let mut rng = StdRng::from_os_rng();
    for _ in 0..100 {
        let n = rng.random_range(1..10);
        run((0..n)
            .map(|_| {
                let num_vars = rng.random_range(0..12);
                let width = rng.random_range(2..5);
                let air = match rng.random_bool(0.5) {
                    false => MyAir::GrandSum { width },
                    true => MyAir::GrandProduct { width },
                };
                let (trace, output) = air.generate_trace_rows(num_vars, &mut rng);
                let public_values = vec![output];
                ProverInput::new(air, public_values, trace)
            })
            .collect());
    }
}
