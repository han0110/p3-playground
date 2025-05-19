use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use itertools::{Itertools, chain, izip};
use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use tracing::instrument;

use crate::{
    PcsError, Proof, StarkGenericConfig, Val, VerifierConstraintFolder, VerifierInput,
    VerifyingKey, eval_log_up,
};

#[instrument(skip_all)]
pub fn verify<SC, A>(
    config: &SC,
    vk: &VerifyingKey,
    inputs: Vec<VerifierInput<Val<SC>, A>>,
    proof: &Proof<SC>,
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let Proof {
        commitments,
        opening_proof,
        ..
    } = proof;

    let valid_shape = proof.per_air.len() == inputs.len()
        && izip!(vk.per_air(), &proof.per_air).all(|(vk, proof)| vk.valid_shape(proof))
        && (vk.has_any_interaction() == commitments.log_up_chunks.is_some());
    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    let (main_domains, quotient_chunks_domains) = izip!(vk.per_air(), &proof.per_air)
        .map(|(vk, proof)| {
            let main_domain = pcs.natural_domain_for_degree(1 << proof.log_degree);
            let quotient_domain = main_domain
                .create_disjoint_domain(1 << (proof.log_degree + vk.log_quotient_degree));
            let quotient_chunks_domains = quotient_domain.split_domains(vk.quotient_degree());
            (main_domain, quotient_chunks_domains)
        })
        .collect::<(Vec<_>, Vec<_>)>();

    // Observe the instance.
    proof
        .per_air
        .iter()
        .for_each(|proof| challenger.observe(Val::<SC>::from_u8(proof.log_degree as u8)));
    // TODO: Might be best practice to include other instance data here in the transcript, like some
    // encoding of the AIR. This protects against transcript collisions between distinct instances.
    // Practically speaking though, the only related known attack is from failing to include public
    // values. It's not clear if failing to include other instance data could enable a transcript
    // collision, since most such changes would completely change the set of satisfying witnesses.

    challenger.observe(commitments.main.clone());
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(&input.public_values));

    let beta: SC::Challenge = challenger.sample_algebra_element();
    let gamma: SC::Challenge = challenger.sample_algebra_element();

    if proof
        .per_air
        .iter()
        .flat_map(|proof| proof.log_up_sum)
        .sum::<SC::Challenge>()
        != SC::Challenge::ZERO
    {
        return Err(VerificationError::InvalidLogUpSum);
    }

    proof
        .per_air
        .iter()
        .flat_map(|proof| proof.log_up_sum)
        .for_each(|sum| challenger.observe_algebra_element(sum));

    if let Some(log_up_chunks) = &commitments.log_up_chunks {
        challenger.observe(log_up_chunks.clone());
    }

    let alpha: SC::Challenge = challenger.sample_algebra_element();

    challenger.observe(commitments.quotient_chunks.clone());

    let zeta: SC::Challenge = challenger.sample();

    pcs.verify(
        chain![
            Some((
                commitments.main.clone(),
                izip!(&main_domains, &proof.per_air)
                    .map(|(main_domain, proof)| {
                        let zeta_next = main_domain.next_point(zeta).unwrap();
                        (
                            *main_domain,
                            vec![
                                (zeta, proof.opened_values.main_local.clone()),
                                (zeta_next, proof.opened_values.main_next.clone()),
                            ],
                        )
                    })
                    .collect(),
            )),
            commitments.log_up_chunks.clone().map(|log_up_chunks| {
                (
                    log_up_chunks,
                    izip!(vk.per_air(), &main_domains, &proof.per_air)
                        .filter(|(vk, _, _)| vk.has_interaction())
                        .map(|(_, main_domain, proof)| {
                            let zeta_next = main_domain.next_point(zeta).unwrap();
                            (
                                *main_domain,
                                vec![
                                    (zeta, proof.opened_values.log_up_local.clone()),
                                    (zeta_next, proof.opened_values.log_up_next.clone()),
                                ],
                            )
                        })
                        .collect(),
                )
            }),
            Some((
                commitments.quotient_chunks.clone(),
                izip!(&quotient_chunks_domains, &proof.per_air)
                    .flat_map(|(quotient_chunks_domains, proof)| {
                        izip!(
                            quotient_chunks_domains,
                            &proof.opened_values.quotient_chunks
                        )
                        .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
                    })
                    .collect(),
            )),
        ]
        .collect(),
        opening_proof,
        &mut challenger,
    )
    .map_err(VerificationError::InvalidOpeningArgument)?;

    let beta_powers = beta
        .powers()
        .skip(1)
        .take(vk.max_field_count())
        .collect_vec();
    let gamma_powers = gamma
        .powers()
        .skip(1)
        .take(vk.max_bus_index() + 1)
        .collect_vec();

    izip!(
        vk.per_air(),
        &inputs,
        main_domains,
        quotient_chunks_domains,
        &proof.per_air
    )
    .try_for_each(|(vk, input, main_domain, quotient_chunks_domains, proof)| {
        let zps = quotient_chunks_domains
            .iter()
            .enumerate()
            .map(|(i, domain)| {
                quotient_chunks_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other_domain)| {
                        other_domain.vanishing_poly_at_point(zeta)
                            * other_domain
                                .vanishing_poly_at_point(domain.first_point())
                                .inverse()
                    })
                    .product::<SC::Challenge>()
            })
            .collect_vec();

        let quotient = proof
            .opened_values
            .quotient_chunks
            .iter()
            .enumerate()
            .map(|(ch_i, ch)| {
                zps[ch_i]
                    * ch.iter()
                        .enumerate()
                        .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i).unwrap() * c)
                        .sum::<SC::Challenge>()
            })
            .sum::<SC::Challenge>();

        let sels = main_domain.selectors_at_point(zeta);

        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(&proof.opened_values.main_local),
            RowMajorMatrixView::new_row(&proof.opened_values.main_next),
        );

        let mut folder = VerifierConstraintFolder {
            main,
            public_values: &input.public_values,
            is_first_row: sels.is_first_row,
            is_last_row: sels.is_last_row,
            is_transition: sels.is_transition,
            alpha,
            accumulator: SC::Challenge::ZERO,
            beta_powers: &beta_powers,
            gamma_powers: &gamma_powers,
            numers: vec![Default::default(); vk.interaction_count],
            denoms: vec![Default::default(); vk.interaction_count],
            interaction_index: 0,
        };
        input.air.eval(&mut folder);

        if let Some(log_up_sum) = proof.log_up_sum {
            let numers = mem::take(&mut folder.numers);
            let denoms = mem::take(&mut folder.denoms);
            let [log_up_local, log_up_next] = [
                &proof.opened_values.log_up_local,
                &proof.opened_values.log_up_next,
            ]
            .map(|values| {
                values
                    .chunks(SC::Challenge::DIMENSION)
                    .map(|ch| {
                        ch.iter()
                            .enumerate()
                            .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i).unwrap() * c)
                            .sum::<SC::Challenge>()
                    })
                    .collect_vec()
            });
            eval_log_up(
                &mut folder,
                &vk.interaction_chunks,
                &numers,
                &denoms,
                &log_up_local,
                &log_up_next,
                log_up_sum,
            );
        }

        let folded_constraints = folder.accumulator;

        // Finally, check that
        //     folded_constraints(zeta) / Z_H(zeta) = quotient(zeta)
        if folded_constraints * sels.inv_vanishing != quotient {
            return Err(VerificationError::OodEvaluationMismatch);
        }

        Ok(())
    })
}

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidProofShape,
    InvalidLogUpSum,
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,
}
