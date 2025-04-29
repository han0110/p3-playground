use core::fmt::Debug;

use p3_air::{Air, BaseAirWithPublicValues};
use p3_air_ext::{SymbolicAirBuilder, SymbolicExpression};
use p3_field::Field;

#[derive(Debug, Clone, Copy)]
pub struct AirMeta {
    pub width: usize,
    pub univariate_degree: usize,
    pub multivariate_degree: usize,
    pub constraint_count: usize,
    pub public_value_count: usize,
}

impl AirMeta {
    pub fn new<Val, A>(air: &A) -> Self
    where
        Val: Field,
        A: Air<SymbolicAirBuilder<Val>> + BaseAirWithPublicValues<Val>,
    {
        let width = air.width();
        let public_value_count = air.num_public_values();
        let univariate_degree = {
            let mut builder = SymbolicAirBuilder::new(0, 0, air.width(), air.num_public_values());
            air.eval(&mut builder);
            let (constraints, _) = builder.into_symbolic_constraints();
            max_degree(&constraints)
        };
        let (multivariate_degree, constraint_count) = {
            let mut builder = SymbolicAirBuilder::new(1, 0, air.width(), air.num_public_values());
            air.eval(&mut builder);
            let (constraints, _) = builder.into_symbolic_constraints();
            (max_degree(&constraints), constraints.len())
        };
        Self {
            width,
            univariate_degree,
            multivariate_degree,
            constraint_count,
            public_value_count,
        }
    }
}

pub(crate) fn max_degree<F>(exprs: &[SymbolicExpression<F>]) -> usize {
    itertools::max(exprs.iter().map(SymbolicExpression::degree_multiple)).unwrap_or(0)
}
