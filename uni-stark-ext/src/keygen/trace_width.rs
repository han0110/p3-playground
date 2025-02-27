use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::{DynamicAir, StarkGenericConfig};

/// Describes the widths of different sections of a trace matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceWidth {
    /// Widths of cached main trace segments.
    pub cached_mains: Vec<usize>,
    /// The common width of the main trace.
    pub common_main: usize,
    /// Width counted in extension field elements, not base field elements.
    pub after_challenge: Vec<usize>,
}

impl TraceWidth {
    /// Constructs `TraceWidth` from an AIR.
    pub(crate) fn from_air<SC, A>(air: &A) -> Self
    where
        SC: StarkGenericConfig,
        A: DynamicAir<SC>,
    {
        Self {
            cached_mains: air.cached_main_widths(),
            common_main: air.common_main_width(),
            after_challenge: Vec::new(),
        }
    }
}
