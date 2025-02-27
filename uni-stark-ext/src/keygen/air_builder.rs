use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, BaseAir};

use super::verifying_params::StarkVerifyingParams;
use crate::trace_width::TraceWidth;
use crate::{
    DynamicAir, StarkGenericConfig, SymbolicAirBuilder, SymbolicExpression, Val,
    get_log_quotient_degree, get_max_constraint_degree,
};

/// A verifying key for a single STARK instance.
#[derive(Clone)]
pub struct StarkVerifyingKey<Val> {
    /// Parameters describing the STARK verification process.
    pub params: StarkVerifyingParams,
    /// The symbolic constraints defining the AIR.
    pub symbolic_constraints: Vec<SymbolicExpression<Val>>,
    /// The factor used to compute the quotient polynomial degree.
    pub quotient_degree: u8,
}

/// A proving key for a single STARK instance.
pub struct StarkProvingKey<SC: StarkGenericConfig> {
    /// The AIR's name for identification.
    pub air_name: String,
    /// The corresponding verifying key.
    pub vk: StarkVerifyingKey<Val<SC>>,
}

/// A builder for generating STARK proving and verifying keys.
pub struct AirKeygenBuilder<SC, A>
where
    SC: StarkGenericConfig,
    A: DynamicAir<SC>,
{
    /// The AIR instance associated with this builder.
    air: Arc<A>,
    /// The width of the execution trace.
    trace_width: TraceWidth,
    /// Marker to retain SC as a type parameter.
    _phantom: PhantomData<SC>,
}

impl<SC, A> AirKeygenBuilder<SC, A>
where
    SC: StarkGenericConfig,
    A: DynamicAir<SC>,
{
    /// Creates a new key generator for an AIR.
    pub fn new(air: Arc<A>) -> Self {
        Self {
            trace_width: TraceWidth::from_air(air.as_ref()),
            air,
            _phantom: PhantomData,
        }
    }

    /// Returns the maximum constraint degree for this AIR.
    #[inline(always)]
    pub(crate) fn max_constraint_degree(&self) -> usize {
        get_max_constraint_degree(self.air.as_ref(), 0, self.air.num_public_values())
    }

    /// Constructs a symbolic representation of the AIR for constraint generation.
    #[inline(always)]
    pub(crate) fn symbolic_builder(&self) -> SymbolicAirBuilder<Val<SC>> {
        let mut builder = SymbolicAirBuilder::new(
            0,
            <A as BaseAir<Val<SC>>>::width(self.air.as_ref()),
            self.air.num_public_values(),
        );
        Air::eval(self.air.as_ref(), &mut builder);
        builder
    }

    /// Computes the quotient polynomial degree for this AIR.
    #[inline(always)]
    fn quotient_degree(&self) -> u8 {
        1 << get_log_quotient_degree(self.air.as_ref(), 0, self.air.num_public_values())
    }

    /// Consumes the builder and generates a proving key.
    pub(crate) fn generate_pk(self) -> StarkProvingKey<SC> {
        let symbolic_builder = self.symbolic_builder();

        let vk = StarkVerifyingKey {
            quotient_degree: self.quotient_degree(),
            params: StarkVerifyingParams::from_width_and_sbuilder(
                self.trace_width,
                &symbolic_builder,
            ),
            symbolic_constraints: symbolic_builder.constraints(),
        };

        StarkProvingKey {
            air_name: self.air.name(),
            vk,
        }
    }
}
