use itertools::{Itertools, chain};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues};
use p3_challenger::{FieldChallenger, HashChallenger, SerializingChallenger32};
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_hyperplonk::{meta, prove, verify};
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

struct MulAir {
    degree: usize,
}

impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        self.degree - 1
    }
}

impl<F> BaseAirWithPublicValues<F> for MulAir {
    fn num_public_values(&self) -> usize {
        1
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for MulAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let is_first_row = builder.is_first_row();
        builder.when_first_row().assert_one(is_first_row);

        let is_last_row = builder.is_last_row();
        builder.when_last_row().assert_one(is_last_row);

        let is_transition = builder.is_transition();
        builder.when_transition().assert_one(is_transition);

        let output = builder.public_values()[0];
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        builder.when_transition().assert_eq(
            local.iter().copied().map_into().product::<AB::Expr>(),
            next[0],
        );
        builder.when_last_row().assert_eq(
            local.iter().copied().map_into().product::<AB::Expr>(),
            output,
        );
    }
}

impl MulAir {
    fn generate_trace_rows<F: Field>(
        &self,
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (RowMajorMatrix<F>, F)
    where
        StandardUniform: Distribution<F>,
    {
        let mut output = F::ONE;
        let input = RowMajorMatrix::new(
            (0..1 << num_vars)
                .flat_map(|_| {
                    let row =
                        chain![[output], (0..self.degree - 2).map(|_| rng.random())].collect_vec();
                    output = row.iter().copied().product();
                    row
                })
                .collect(),
            BaseAir::<F>::width(self),
        );
        (input, output)
    }
}

fn run_mul_air(degree: usize) {
    let mut rng = StdRng::from_os_rng();

    let air = MulAir { degree };
    assert_eq!(meta::<Val, _>(&air).1, degree);

    for num_vars in 0..12 {
        let (input, output) = air.generate_trace_rows(num_vars, &mut rng);
        let public_values = vec![output];

        let mut prover_challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
        let proof = prove(&air, &public_values, input.clone(), &mut prover_challenger);

        let mut verifier_challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash {});
        verify::<_, Challenge, _>(&air, &public_values, &proof, &mut verifier_challenger).unwrap();

        assert_eq!(
            prover_challenger.sample_algebra_element::<Challenge>(),
            verifier_challenger.sample_algebra_element::<Challenge>(),
        );
    }
}

#[test]
fn mul_air() {
    for degree in 2..5 {
        run_mul_air(degree);
    }
}
