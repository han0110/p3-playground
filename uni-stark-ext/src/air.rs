use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, BaseAir, BaseAirWithPublicValues};

use crate::{StarkGenericConfig, SymbolicAirBuilder, Val};

/// A dynamically dispatchable AIR supporting multiple base traits.
pub trait DynamicAir<SC: StarkGenericConfig>:
    Air<SymbolicAirBuilder<Val<SC>>> + BaseAirWithPublicValues<Val<SC>> + PartitionedBaseAir<Val<SC>>
{
    /// Returns the name of the AIR for display purposes.
    fn name(&self) -> String;
}

/// Defines an AIR with one or more main trace partitions.
pub trait PartitionedBaseAir<F>: BaseAir<F> {
    /// Returns the widths of cached main trace partitions, if any.
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![]
    }

    /// Returns the width of the common private main trace.
    fn common_main_width(&self) -> usize {
        self.width()
    }
}
