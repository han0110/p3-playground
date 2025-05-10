use itertools::izip;
use p3_air::{AirBuilder, AirBuilderWithPublicValues};
use p3_air_ext::{InteractionAirBuilder, InteractionType};
use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;

pub type ProverInteractionFolderOnExtension<'a, F, EF> =
    ProverInteractionFolderGeneric<'a, F, EF, EF, EF>;

pub type ProverInteractionFolderOnPacking<'a, F, EF> = ProverInteractionFolderGeneric<
    'a,
    F,
    EF,
    <F as Field>::Packing,
    <EF as ExtensionField<F>>::ExtensionPacking,
>;

pub struct ProverInteractionFolderGeneric<'a, F, EF, Var, VarEF> {
    pub main: RowMajorMatrixView<'a, Var>,
    pub public_values: &'a [F],
    pub beta_powers: &'a [EF],
    pub gamma_powers: &'a [EF],
    pub numers: &'a mut [Var],
    pub denoms: &'a mut [VarEF],
    pub interaction_index: usize,
}

impl<'a, F, EF, Var, VarEF> AirBuilder for ProverInteractionFolderGeneric<'a, F, EF, Var, VarEF>
where
    F: Field,
    EF: ExtensionField<F>,
    Var: Algebra<F> + Copy + Send + Sync,
    VarEF: Algebra<Var>,
{
    type F = F;
    type Expr = Var;
    type Var = Var;
    type M = RowMajorMatrixView<'a, Var>;

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

impl<F, EF, Var, VarEF> AirBuilderWithPublicValues
    for ProverInteractionFolderGeneric<'_, F, EF, Var, VarEF>
where
    F: Field,
    EF: ExtensionField<F>,
    Var: Algebra<F> + Copy + Send + Sync,
    VarEF: Algebra<Var>,
{
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<F, EF, Var, VarEF> InteractionAirBuilder
    for ProverInteractionFolderGeneric<'_, F, EF, Var, VarEF>
where
    F: Field,
    EF: ExtensionField<F>,
    Var: Algebra<F> + Copy + Send + Sync,
    VarEF: Algebra<Var> + From<EF>,
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
        let mut numer = count.into();
        if interaction_type == InteractionType::Receive {
            numer = -numer;
        }
        self.numers[self.interaction_index] = numer;

        let denom = {
            let mut fields = fields.into_iter();
            VarEF::from(self.gamma_powers[bus_index])
                + fields.next().unwrap().into()
                + izip!(fields, self.beta_powers)
                    .map(|(field, beta_power)| VarEF::from(*beta_power) * field.into())
                    .sum::<VarEF>()
        };
        self.denoms[self.interaction_index] = denom;

        self.interaction_index += 1;
    }
}
