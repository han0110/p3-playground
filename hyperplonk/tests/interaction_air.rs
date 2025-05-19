use core::cmp::max;
use core::iter::Product;

use itertools::{Itertools, cloned};
use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_air_ext::{InteractionBuilder, ProverInput};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Algebra, Field, PrimeCharacteristicRing};
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
struct SendingAir {
    constraint_degree: usize,
    interaction_degree: usize,
}

impl<F> BaseAir<F> for SendingAir {
    fn width(&self) -> usize {
        max(self.constraint_degree + 1, self.interaction_degree)
    }
}

impl<AB: InteractionBuilder> Air<AB> for SendingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();
        if !AB::ONLY_INTERACTION {
            let (output, inputs) = local.split_last().unwrap();
            builder.assert_eq(
                AB::Expr::product(inputs.iter().copied().map_into()),
                *output,
            );
        }
        builder.push_send(0, self.interaction_values(&local, &next), AB::Expr::ONE);
        builder.push_send(1, [AB::Expr::ONE], AB::Expr::ONE);
        builder.push_receive(1, [AB::Expr::ONE], AB::Expr::ONE);
    }
}

#[derive(Clone, Copy)]
struct ReceivingAir;

impl<F> BaseAir<F> for ReceivingAir {
    fn width(&self) -> usize {
        5 // [..values, mult]
    }
}

impl<AB: InteractionBuilder> Air<AB> for ReceivingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        let (mult, values) = local.split_last().unwrap();
        builder.push_receive(0, cloned(values), *mult);
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

impl<AB: InteractionBuilder> Air<AB> for MyAir {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Sending(inner) => inner.eval(builder),
            Self::Receiving(inner) => inner.eval(builder),
        }
    }
}

impl SendingAir {
    fn interaction_values<Var: Copy + Into<Expr>, Expr: Algebra<Var>>(
        &self,
        local: &[Var],
        next: &[Var],
    ) -> [Expr; 4] {
        [
            Expr::sum(local.iter().copied().map_into()),
            Expr::sum(next.iter().copied().map_into()),
            Expr::product(local[..self.interaction_degree].iter().copied().map_into()),
            Expr::product(next[..self.interaction_degree].iter().copied().map_into()),
        ]
    }

    fn generate_sending_trace<F: Field>(&self, n: usize, mut rng: impl Rng) -> RowMajorMatrix<F> {
        let mut trace = RowMajorMatrix::new(
            F::zero_vec(n * BaseAir::<F>::width(self)),
            BaseAir::<F>::width(self),
        );
        trace.rows_mut().for_each(|row| {
            let (output, inputs) = row.split_last_mut().unwrap();
            inputs
                .iter_mut()
                .for_each(|input| *input = F::from_u8(rng.random::<u8>() & 0b11));
            *output = inputs.iter().copied().product();
        });
        trace
    }

    fn generate_receiving_trace<F: Field + Ord>(
        &self,
        sending_trace: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<F> {
        let values = (0..sending_trace.height())
            .map(|row| {
                let local = sending_trace.row_slice(row).unwrap();
                let next = sending_trace
                    .row_slice((row + 1) % sending_trace.height())
                    .unwrap();
                self.interaction_values::<F, F>(&local, &next)
            })
            .collect_vec();
        let counts = values
            .into_iter()
            .counts()
            .into_iter()
            .sorted_by_key(|(key, _)| *key);
        let mut trace = RowMajorMatrix::new(F::zero_vec(5 * counts.len().next_power_of_two()), 5);
        trace
            .rows_mut()
            .zip(counts)
            .for_each(|(row, (values, mult))| {
                row[..4].copy_from_slice(&values);
                row[4] = F::from_usize(mult);
            });
        trace
    }
}

#[test]
fn interaction() {
    let mut rng = StdRng::from_os_rng();
    for ((log_b, constraint_degree), interaction_degree) in
        (0..12).cartesian_product(0..4).cartesian_product(0..4)
    {
        let sending_air = SendingAir {
            constraint_degree,
            interaction_degree,
        };
        let sending_trace = sending_air.generate_sending_trace(1 << log_b, &mut rng);
        let receiving_trace = sending_air.generate_receiving_trace(&sending_trace);

        run::<Val, Challenge, _>(vec![
            ProverInput::new(MyAir::Sending(sending_air), Vec::new(), sending_trace),
            ProverInput::new(MyAir::Receiving(ReceivingAir), Vec::new(), receiving_trace),
        ]);
    }
}
