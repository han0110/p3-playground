use itertools::{Itertools, izip};
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn seeded_rng() -> impl Rng {
    StdRng::seed_from_u64(0)
}

fn do_test_uni_whir_pcs<Val, Challenge, Challenger, P>(
    (pcs, challenger): &(P, Challenger),
    log_degrees_by_round: &[&[usize]],
) where
    P: Pcs<Challenge, Challenger>,
    P::Domain: PolynomialSpace<Val = Val>,
    Val: Field,
    StandardUniform: Distribution<Val>,
    Challenge: ExtensionField<Val>,
    Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
{
    let num_rounds = log_degrees_by_round.len();
    let mut rng = seeded_rng();

    let mut p_challenger = challenger.clone();

    let domains_and_polys_by_round = log_degrees_by_round
        .iter()
        .map(|log_degrees| {
            log_degrees
                .iter()
                .map(|&log_degree| {
                    let d = 1 << log_degree;
                    // random width 5-15
                    let width = 5 + rng.random_range(0..=10);
                    (
                        pcs.natural_domain_for_degree(d),
                        RowMajorMatrix::<Val>::rand(&mut rng, d, width),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let (commits_by_round, data_by_round): (Vec<_>, Vec<_>) = domains_and_polys_by_round
        .iter()
        .map(|domains_and_polys| pcs.commit(domains_and_polys.clone()))
        .unzip();
    assert_eq!(commits_by_round.len(), num_rounds);
    assert_eq!(data_by_round.len(), num_rounds);
    p_challenger.observe_slice(&commits_by_round);

    let zeta: Challenge = p_challenger.sample_algebra_element();

    let points_by_round = log_degrees_by_round
        .iter()
        .map(|log_degrees| vec![vec![zeta]; log_degrees.len()])
        .collect_vec();
    let data_and_points = data_by_round.iter().zip(points_by_round).collect();
    let (opening_by_round, proof) = pcs.open(data_and_points, &mut p_challenger);
    assert_eq!(opening_by_round.len(), num_rounds);

    // Verify the proof.
    let mut v_challenger = challenger.clone();
    v_challenger.observe_slice(&commits_by_round);
    let verifier_zeta: Challenge = v_challenger.sample_algebra_element();
    assert_eq!(verifier_zeta, zeta);

    let commits_and_claims_by_round = izip!(
        commits_by_round,
        domains_and_polys_by_round,
        opening_by_round
    )
    .map(|(commit, domains_and_polys, openings)| {
        let claims = domains_and_polys
            .iter()
            .zip(openings)
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect_vec();
        (commit, claims)
    })
    .collect_vec();
    assert_eq!(commits_and_claims_by_round.len(), num_rounds);

    pcs.verify(commits_and_claims_by_round, &proof, &mut v_challenger)
        .unwrap()
}

// Set it up so we create tests inside a module for each pcs, so we get nice error reports
// specific to a failing PCS.
macro_rules! make_tests_for_pcs {
    ($p:expr) => {
        #[test]
        fn single() {
            let p = $p;
            for i in 3..6 {
                $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[i]]);
            }
        }

        #[test]
        fn many_equal() {
            let p = $p;
            for i in 5..8 {
                $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[i; 5]]);
            }
        }

        #[test]
        fn many_different() {
            let p = $p;
            for i in 3..8 {
                let degrees = (3..3 + i).collect::<Vec<_>>();
                $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&degrees]);
            }
        }

        #[test]
        fn many_different_rev() {
            let p = $p;
            for i in 3..8 {
                let degrees = (3..3 + i).rev().collect::<Vec<_>>();
                $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&degrees]);
            }
        }

        #[test]
        fn multiple_rounds() {
            let p = $p;
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[3]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[3], &[3]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[3], &[2]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[2], &[3]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[3, 4], &[3, 4]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[4, 2], &[4, 2]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[2, 2], &[3, 3]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[3, 3], &[2, 2]]);
            $crate::do_test_uni_whir_pcs::<_, super::Challenge, _, _>(&p, &[&[2], &[3, 3]]);
        }
    };
}

mod babybear_uni_whir_pcs {
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
    use p3_uni_whir::{FoldType, FoldingFactor, ProtocolParameters, SoundnessType, UniWhirPcs};

    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type ByteHash = Keccak256Hash;
    type MyHash = SerializingHasher32<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

    type Dft = Radix2DitParallel<Val>;

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type MyPcs = UniWhirPcs<Val, Dft, MyHash, MyCompress, 32>;

    fn get_pcs(
        log_blowup: usize,
        folding_factor: usize,
        first_round_folding_factor: usize,
    ) -> (MyPcs, Challenger) {
        let dft = Dft::default();

        let security_level = 100;
        let pow_bits = 20;

        let byte_hash = ByteHash {};
        let field_hash = MyHash::new(byte_hash);

        let compress = MyCompress::new(byte_hash);

        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            merkle_hash: field_hash,
            merkle_compress: compress,
            soundness_type: SoundnessType::ConjectureList,
            fold_optimisation: FoldType::ProverHelps,
            starting_log_inv_rate: log_blowup,
        };

        (
            MyPcs::new(dft, whir_params),
            Challenger::from_hasher(Vec::new(), byte_hash),
        )
    }

    mod blowup_1 {
        make_tests_for_pcs!(super::get_pcs(1, 4, 4));
    }

    mod blowup_2 {
        make_tests_for_pcs!(super::get_pcs(2, 4, 4));
    }
}
