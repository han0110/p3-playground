use alloc::rc::Rc;
use core::cmp;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Algebra, Field, InjectiveMonomial, PrimeCharacteristicRing};

use crate::symbolic_variable::SymbolicVariable;

/// An expression over `SymbolicVariable`s.
#[derive(Clone, Debug)]
pub enum SymbolicExpression<F> {
    Variable(SymbolicVariable<F>),
    IsFirstRow,
    IsLastRow,
    IsTransition {
        degree: usize,
    },
    Constant(F),
    Add {
        x: Rc<Self>,
        y: Rc<Self>,
        degree_multiple: usize,
    },
    Sub {
        x: Rc<Self>,
        y: Rc<Self>,
        degree_multiple: usize,
    },
    Neg {
        x: Rc<Self>,
        degree_multiple: usize,
    },
    Mul {
        x: Rc<Self>,
        y: Rc<Self>,
        degree_multiple: usize,
    },
}

impl<F> SymbolicExpression<F> {
    /// Returns the multiple of `n` (the trace length) in this expression's degree.
    pub const fn degree_multiple(&self) -> usize {
        match self {
            Self::Variable(v) => v.degree_multiple(),
            Self::IsFirstRow | Self::IsLastRow => 1,
            Self::IsTransition { degree } => *degree,
            Self::Constant(_) => 0,
            Self::Add {
                degree_multiple, ..
            }
            | Self::Sub {
                degree_multiple, ..
            }
            | Self::Neg {
                degree_multiple, ..
            }
            | Self::Mul {
                degree_multiple, ..
            } => *degree_multiple,
        }
    }

    pub fn has_selector(&self) -> bool {
        match self {
            Self::Variable(_) | Self::Constant(_) => false,
            Self::IsFirstRow | Self::IsLastRow | Self::IsTransition { .. } => true,
            Self::Neg { x, .. } => x.has_selector(),
            Self::Add { x, y, .. } | Self::Sub { x, y, .. } | Self::Mul { x, y, .. } => {
                x.has_selector() || y.has_selector()
            }
        }
    }
}

impl<F: Field> Default for SymbolicExpression<F> {
    fn default() -> Self {
        Self::Constant(F::ZERO)
    }
}

impl<F: Field> From<F> for SymbolicExpression<F> {
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<F: Field> PrimeCharacteristicRing for SymbolicExpression<F> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        F::from_prime_subfield(f).into()
    }
}

impl<F: Field> Algebra<F> for SymbolicExpression<F> {}

impl<F: Field> Algebra<SymbolicVariable<F>> for SymbolicExpression<F> {}

// Note we cannot implement PermutationMonomial due to the degree_multiple part which makes
// operations non invertible.
impl<F: Field + InjectiveMonomial<N>, const N: u64> InjectiveMonomial<N> for SymbolicExpression<F> {}

impl<F: Field, T> Add<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        let rhs = rhs.into();
        match (self, rhs) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs + rhs),
            (lhs, rhs) => {
                let degree_multiple = cmp::max(lhs.degree_multiple(), rhs.degree_multiple());
                Self::Add {
                    x: Rc::new(lhs),
                    y: Rc::new(rhs),
                    degree_multiple,
                }
            }
        }
    }
}

impl<F: Field, T> AddAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into();
    }
}

impl<F: Field, T> Sum<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x + y)
            .unwrap_or(Self::ZERO)
    }
}

impl<F: Field, T> Sub<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        let rhs = rhs.into();
        match (self, rhs) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs - rhs),
            (lhs, rhs) => {
                let degree_multiple = cmp::max(lhs.degree_multiple(), rhs.degree_multiple());
                Self::Sub {
                    x: Rc::new(lhs),
                    y: Rc::new(rhs),
                    degree_multiple,
                }
            }
        }
    }
}

impl<F: Field, T> SubAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into();
    }
}

impl<F: Field> Neg for SymbolicExpression<F> {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Constant(c) => Self::Constant(-c),
            expr => {
                let degree_multiple = expr.degree_multiple();
                Self::Neg {
                    x: Rc::new(expr),
                    degree_multiple,
                }
            }
        }
    }
}

impl<F: Field, T> Mul<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let rhs = rhs.into();
        match (self, rhs) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs * rhs),
            (lhs, rhs) => {
                #[allow(clippy::suspicious_arithmetic_impl)]
                let degree_multiple = lhs.degree_multiple() + rhs.degree_multiple();
                Self::Mul {
                    x: Rc::new(lhs),
                    y: Rc::new(rhs),
                    degree_multiple,
                }
            }
        }
    }
}

impl<F: Field, T> MulAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs.into();
    }
}

impl<F: Field, T> Product<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x * y)
            .unwrap_or(Self::ONE)
    }
}
