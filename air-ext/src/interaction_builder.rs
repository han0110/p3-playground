use alloc::vec::Vec;

use p3_air::AirBuilder;
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

pub trait InteractionBuilder: AirBuilder {
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
