use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_field::ExtensionField;
use p3_ml_pcs::MlPcs;

pub type PcsError<SC> = <<SC as HyperPlonkGenericConfig>::Pcs as MlPcs<
    <SC as HyperPlonkGenericConfig>::Challenge,
    <SC as HyperPlonkGenericConfig>::Challenger,
>>::Error;

pub type Val<C> = <<C as HyperPlonkGenericConfig>::Pcs as MlPcs<
    <C as HyperPlonkGenericConfig>::Challenge,
    <C as HyperPlonkGenericConfig>::Challenger,
>>::Val;

pub trait HyperPlonkGenericConfig {
    /// The PCS used to commit to trace polynomials.
    type Pcs: MlPcs<Self::Challenge, Self::Challenger>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Val<Self>>;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanObserve<<Self::Pcs as MlPcs<Self::Challenge, Self::Challenger>>::Commitment>
        + CanSample<Self::Challenge>;

    /// Get a reference to the PCS used by this proof configuration.
    fn pcs(&self) -> &Self::Pcs;

    /// Get an initialisation of the challenger used by this proof configuration.
    fn initialise_challenger(&self) -> Self::Challenger;
}

#[derive(Debug)]
pub struct HyperPlonkConfig<Pcs, Challenge, Challenger> {
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
    /// An initialised instance of the challenger.
    challenger: Challenger,
    _phantom: PhantomData<Challenge>,
}

impl<Pcs, Challenge, Challenger> HyperPlonkConfig<Pcs, Challenge, Challenger> {
    pub const fn new(pcs: Pcs, challenger: Challenger) -> Self {
        Self {
            pcs,
            challenger,
            _phantom: PhantomData,
        }
    }
}

impl<Pcs, Challenge, Challenger> HyperPlonkGenericConfig
    for HyperPlonkConfig<Pcs, Challenge, Challenger>
where
    Challenge: ExtensionField<Pcs::Val>,
    Pcs: MlPcs<Challenge, Challenger>,
    Challenger: FieldChallenger<Pcs::Val>
        + CanObserve<<Pcs as MlPcs<Challenge, Challenger>>::Commitment>
        + CanSample<Challenge>
        + Clone,
{
    type Pcs = Pcs;
    type Challenge = Challenge;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        self.challenger.clone()
    }
}
