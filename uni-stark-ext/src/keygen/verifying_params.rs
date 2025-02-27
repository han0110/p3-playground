use p3_field::Field;
use serde::{Deserialize, Serialize};

use super::trace_width::TraceWidth;
use crate::SymbolicAirBuilder;

/// Verification parameters for a STARK proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarkVerifyingParams {
    /// Describes the widths of different trace sub-matrices.
    pub width: TraceWidth,
    /// The number of public values used in this STARK instance.
    pub num_public_values: usize,
}

impl StarkVerifyingParams {
    /// Constructs `StarkVerifyingParams` from trace width and a symbolic builder.
    pub fn from_width_and_sbuilder<F: Field>(
        width: TraceWidth,
        symbolic_builder: &SymbolicAirBuilder<F>,
    ) -> Self {
        Self {
            width,
            num_public_values: symbolic_builder.public_values.len(),
        }
    }
}
