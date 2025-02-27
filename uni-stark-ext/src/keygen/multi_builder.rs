use alloc::sync::Arc;
use alloc::vec::Vec;

use super::air_builder::{AirKeygenBuilder, StarkProvingKey};
use super::verifying_params::StarkVerifyingParams;
use crate::{DynamicAir, StarkGenericConfig, SymbolicExpression};

/// A proving key for multiple AIRs.
pub struct MultiStarkProvingKey<SC: StarkGenericConfig> {
    /// Proving keys for each AIR.
    pub per_air: Vec<StarkProvingKey<SC>>,
    /// Maximum constraint degree across all AIRs.
    ///
    /// For now this is useless but we keep it here since it will be used once we have RAP.
    pub max_constraint_degree: usize,
}

/// A verifying key for a single STARK, corresponding to a single AIR matrix.
#[derive(Clone)]
pub struct StarkVerifyingKey<Val> {
    /// STARK verification parameters.
    pub params: StarkVerifyingParams,
    /// Symbolic constraints defining the AIR.
    pub symbolic_constraints: Vec<SymbolicExpression<Val>>,
    /// The quotient polynomial degree factor, derived from the max constraint degree.
    pub quotient_degree: u8,
}

/// A builder for creating MultiStark proving and verifying keys.
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig, A: DynamicAir<SC>> {
    /// Reference to the STARK configuration.
    pub config: &'a SC,
    /// AIRs partitioned by constraints and trace width.
    partitioned_airs: Vec<AirKeygenBuilder<SC, A>>,
    /// The maximum constraint degree allowed across all AIRs.
    max_constraint_degree: usize,
}

impl<'a, SC: StarkGenericConfig, A: DynamicAir<SC>> MultiStarkKeygenBuilder<'a, SC, A> {
    /// Creates a new builder for MultiStark proving keys.
    pub const fn new(config: &'a SC) -> Self {
        Self {
            config,
            partitioned_airs: Vec::new(),
            max_constraint_degree: 0,
        }
    }

    /// Sets the maximum constraint degree for the AIRs in this builder.
    #[inline(always)]
    pub fn set_max_constraint_degree(&mut self, max_constraint_degree: usize) {
        self.max_constraint_degree = max_constraint_degree;
    }

    /// Adds a single interactive AIR and returns its index.
    #[inline(always)]
    pub fn add_air(&mut self, air: Arc<A>) -> usize {
        self.partitioned_airs.push(AirKeygenBuilder::new(air));
        self.partitioned_airs.len() - 1
    }

    /// Adds multiple AIRs at once.
    #[inline(always)]
    pub fn add_airs(&mut self, airs: Vec<Arc<A>>) {
        airs.into_iter().for_each(|air| {
            self.add_air(air);
        });
    }

    /// Consumes the builder and generates the MultiStark proving key.
    pub fn generate_pk(mut self) -> MultiStarkProvingKey<SC> {
        if self.max_constraint_degree != 0 {
            let air_max_constraint_degree = self
                .partitioned_airs
                .iter()
                .map(|keygen_builder| keygen_builder.max_constraint_degree())
                .max()
                .unwrap();

            if air_max_constraint_degree > self.max_constraint_degree {
                self.set_max_constraint_degree(air_max_constraint_degree);
            }
        }

        MultiStarkProvingKey {
            per_air: self
                .partitioned_airs
                .into_iter()
                .map(|keygen_builder| keygen_builder.generate_pk())
                .collect(),
            max_constraint_degree: self.max_constraint_degree,
        }
    }
}
