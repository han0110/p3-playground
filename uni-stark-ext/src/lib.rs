//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod folder;
mod interaction_air_builder;
mod keygen;
mod proof;
mod prover;
mod symbolic_builder;
mod symbolic_expression;
mod symbolic_variable;
mod vanishing_polynomial_coset;
mod verifier;

#[cfg(feature = "check-constraints")]
mod check_constraints;

#[cfg(feature = "check-constraints")]
pub use check_constraints::*;
pub use config::*;
pub use folder::*;
pub use interaction_air_builder::*;
pub use keygen::*;
pub use proof::*;
pub use prover::*;
pub use symbolic_builder::*;
pub use symbolic_expression::*;
pub use symbolic_variable::*;
pub use vanishing_polynomial_coset::*;
pub use verifier::*;
