use core::ops::Deref;

use p3_air::{AirBuilder, AirBuilderWithPublicValues};
use p3_matrix::Matrix;

use crate::{InteractionBuilder, InteractionType};

pub struct SubAirBuilder<'a, AB> {
    inner: &'a mut AB,
    start: usize,
    width: usize,
}

impl<'a, AB> SubAirBuilder<'a, AB> {
    #[inline]
    pub fn new(inner: &'a mut AB, start: usize, width: usize) -> Self {
        Self {
            inner,
            start,
            width,
        }
    }
}

impl<AB: AirBuilder> AirBuilder for SubAirBuilder<'_, AB> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = HorizontallyCropped<AB::M>;

    #[inline]
    fn main(&self) -> Self::M {
        HorizontallyCropped::new(self.inner.main(), self.start, self.width).unwrap()
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x)
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        self.inner.assert_zeros(array)
    }
}

impl<AB: AirBuilderWithPublicValues> AirBuilderWithPublicValues for SubAirBuilder<'_, AB> {
    type PublicVar = AB::PublicVar;

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
    }
}

impl<AB: InteractionBuilder> InteractionBuilder for SubAirBuilder<'_, AB> {
    const ONLY_INTERACTION: bool = AB::ONLY_INTERACTION;

    #[inline]
    fn push_interaction(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item: Into<Self::Expr>>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        self.inner
            .push_interaction(bus_index, fields, count, interaction_type);
    }
}

pub struct HorizontallyCropped<M> {
    inner: M,
    start: usize,
    width: usize,
}

impl<M> HorizontallyCropped<M> {
    #[inline]
    pub fn new<T: Clone + Send + Sync>(inner: M, start: usize, width: usize) -> Option<Self>
    where
        M: Matrix<T>,
    {
        (start + width <= inner.width()).then(|| Self {
            inner,
            start,
            width,
        })
    }
}

impl<M: Matrix<T>, T: Clone + Send + Sync> Matrix<T> for HorizontallyCropped<M> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.height()
    }

    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe { self.inner.get_unchecked(r, self.start + c) }
    }

    #[inline]
    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            self.inner
                .row_subseq_unchecked(r, self.start, self.start + self.width)
        }
    }

    #[inline]
    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            self.inner
                .row_subseq_unchecked(r, self.start + start, self.start + end)
        }
    }

    #[inline]
    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            self.inner
                .row_subslice_unchecked(r, self.start + start, self.start + end)
        }
    }
}
