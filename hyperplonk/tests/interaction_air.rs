use itertools::Itertools;
use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_air_ext::{InteractionAirBuilder, ProverInput};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use util::run;

mod util;

type Val = KoalaBear;
type Challenge = BinomialExtensionField<Val, 4>;

#[derive(Clone, Copy)]
struct SendingAir;

impl<F> BaseAir<F> for SendingAir {
    fn width(&self) -> usize {
        1 // [value]
    }
}

impl<AB: InteractionAirBuilder> Air<AB> for SendingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        if !AB::ONLY_INTERACTION {
            builder.assert_eq(local[0].into().square(), local[0].into().square());
        }
        builder.push_send(0, [local[0]], AB::Expr::ONE);
    }
}

#[derive(Clone, Copy)]
struct ReceivingAir;

impl<F> BaseAir<F> for ReceivingAir {
    fn width(&self) -> usize {
        2 // [value, mult]
    }
}

impl<AB: InteractionAirBuilder> Air<AB> for ReceivingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        if !AB::ONLY_INTERACTION {
            builder.assert_eq(local[0].into().square(), local[0].into().square());
        }
        builder.push_receive(0, [local[0]], local[1]);
    }
}

#[derive(Clone, Copy)]
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

impl<F> BaseAirWithPublicValues<F> for MyAir {}

impl<AB: InteractionAirBuilder> Air<AB> for MyAir {
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

#[test]
fn single_sum() {
    let mut rng = StdRng::from_os_rng();
    for num_vars in 0..12 {
        let sending_trace = generate_sending_trace(1 << num_vars, &mut rng);
        let receiving_trace = generate_receiving_trace(&sending_trace);

        run::<Val, Challenge, _>(vec![
            ProverInput::new(MyAir::Sending(SendingAir), Vec::new(), sending_trace),
            ProverInput::new(MyAir::Receiving(ReceivingAir), Vec::new(), receiving_trace),
        ]);
    }
}
