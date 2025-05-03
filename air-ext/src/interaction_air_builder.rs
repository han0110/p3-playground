use alloc::vec::Vec;

use itertools::izip;
use p3_air::{AirBuilder, AirBuilderWithPublicValues};
use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InteractionType {
    Send,
    Receive,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interaction<Expr> {
    pub fields: Vec<Expr>,
    pub count: Expr,
    pub bus_index: usize,
    pub interaction_type: InteractionType,
}

pub trait InteractionAirBuilder: AirBuilder {
    const ONLY_INTERACTION: bool;

    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    );

    #[inline]
    fn push_send(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Send);
    }

    #[inline]
    fn push_receive(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
    ) {
        self.push_interaction(bus_index, fields, count, InteractionType::Receive);
    }
}

pub type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

pub struct ProverInteractionFolder<
    'a,
    Val,
    Challenge,
    Var = Val,
    VarEF = Challenge,
    M = ViewPair<'a, Var>,
> {
    pub main: M,
    pub public_values: &'a Vec<Val>,
    pub beta_powers: &'a [Challenge],
    pub gamma_powers: &'a [Challenge],
    pub numers: &'a mut [Var],
    pub denoms: &'a mut [VarEF],
    pub interaction_index: usize,
}

impl<Val, Challenge, Var, VarEF, M> AirBuilder
    for ProverInteractionFolder<'_, Val, Challenge, Var, VarEF, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Var: Algebra<Val> + Copy + Send + Sync,
    VarEF: Algebra<Var>,
    M: Copy + Matrix<Var>,
{
    type F = Val;
    type Expr = Var;
    type Var = Var;
    type M = M;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        unimplemented!()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        unimplemented!()
    }

    #[inline]
    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unimplemented!()
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, _: I) {}
}

impl<Val, Challenge, Var, VarEF, M> AirBuilderWithPublicValues
    for ProverInteractionFolder<'_, Val, Challenge, Var, VarEF, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Var: Algebra<Val> + Copy + Send + Sync,
    VarEF: Algebra<Var>,
    M: Copy + Matrix<Var>,
{
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<Val, Challenge, Var, VarEF, M> InteractionAirBuilder
    for ProverInteractionFolder<'_, Val, Challenge, Var, VarEF, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Var: Algebra<Val> + Copy + Send + Sync,
    VarEF: Algebra<Var> + From<Challenge>,
    M: Copy + Matrix<Var>,
{
    const ONLY_INTERACTION: bool = true;

    #[inline]
    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let mut count = count.into();
        if interaction_type == InteractionType::Receive {
            count = -count;
        }
        self.numers[self.interaction_index] = count;

        let mut fields = fields.into_iter();
        self.denoms[self.interaction_index] =
            VarEF::from(self.gamma_powers[bus_index]) + fields.next().unwrap().into();
        izip!(fields, self.beta_powers).for_each(|(field, beta_power)| {
            self.denoms[self.interaction_index] += VarEF::from(*beta_power) * field.into();
        });

        self.interaction_index += 1;
    }
}
