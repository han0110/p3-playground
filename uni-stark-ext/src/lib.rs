//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod folder;
mod interaction_air_builder;
mod keygen;
mod proof;
mod prover;
mod verifier;

pub use config::*;
pub use folder::*;
pub use interaction_air_builder::*;
pub use keygen::*;
pub use p3_air_ext::*;
pub use proof::*;
pub use prover::*;
pub use verifier::*;
