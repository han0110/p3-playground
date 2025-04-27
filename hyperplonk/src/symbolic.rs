use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cmp;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAirWithPublicValues};
use p3_field::{Algebra, Field, InjectiveMonomial, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    Preprocessed { offset: usize },
    Main { offset: usize },
    Permutation { offset: usize },
    Public,
    Challenge,
}

#[derive(Copy, Clone, Debug)]
pub struct SymbolicVariable<F> {
    pub entry: Entry,
    pub index: usize,
    _phantom: PhantomData<F>,
}

impl<F> SymbolicVariable<F> {
    pub const fn new(entry: Entry, index: usize) -> Self {
        Self {
            entry,
            index,
            _phantom: PhantomData,
        }
    }

    pub const fn degree_multiple(&self) -> usize {
        match self.entry {
            Entry::Preprocessed { .. } | Entry::Main { .. } | Entry::Permutation { .. } => 1,
            Entry::Public | Entry::Challenge => 0,
        }
    }
}

impl<F: Field> From<SymbolicVariable<F>> for SymbolicExpression<F> {
    fn from(value: SymbolicVariable<F>) -> Self {
        Self::Variable(value)
    }
}

impl<F: Field, T> Add<T> for SymbolicVariable<F>
where
    T: Into<SymbolicExpression<F>>,
{
    type Output = SymbolicExpression<F>;

    fn add(self, rhs: T) -> Self::Output {
        SymbolicExpression::from(self) + rhs.into()
    }
}

impl<F: Field, T> Sub<T> for SymbolicVariable<F>
where
    T: Into<SymbolicExpression<F>>,
{
    type Output = SymbolicExpression<F>;

    fn sub(self, rhs: T) -> Self::Output {
        SymbolicExpression::from(self) - rhs.into()
    }
}

impl<F: Field, T> Mul<T> for SymbolicVariable<F>
where
    T: Into<SymbolicExpression<F>>,
{
    type Output = SymbolicExpression<F>;

    fn mul(self, rhs: T) -> Self::Output {
        SymbolicExpression::from(self) * rhs.into()
    }
}

#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
    is_transition_degree: usize,
}

impl<F: Field> SymbolicAirBuilder<F> {
    fn new(width: usize, num_public_values: usize, is_transition_degree: usize) -> Self {
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        Self {
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            constraints: Vec::new(),
            is_transition_degree,
        }
    }
}

fn max_degree<F>(exprs: &[SymbolicExpression<F>]) -> usize {
    itertools::max(exprs.iter().map(SymbolicExpression::degree_multiple)).unwrap_or(0)
}

impl<F: Field> AirBuilder for SymbolicAirBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition {
                degree: self.is_transition_degree,
            }
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

impl<F: Field> AirBuilderWithPublicValues for SymbolicAirBuilder<F> {
    type PublicVar = SymbolicVariable<F>;
    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

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
            let mut builder = SymbolicAirBuilder::new(air.width(), air.num_public_values(), 0);
            air.eval(&mut builder);
            max_degree(&builder.constraints)
        };
        let mut builder = SymbolicAirBuilder::new(air.width(), air.num_public_values(), 1);
        air.eval(&mut builder);
        let multivariate_degree = max_degree(&builder.constraints);
        assert!(univariate_degree >= 2, "Not yet supported");
        assert!(multivariate_degree >= 2, "Not yet supported");
        let constraint_count = builder.constraints.len();
        Self {
            width,
            univariate_degree,
            multivariate_degree,
            constraint_count,
            public_value_count,
        }
    }
}
