use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_air::ExtensionBuilder;

#[inline]
pub(crate) fn eval_log_up<AB: ExtensionBuilder>(
    builder: &mut AB,
    interaction_chunks: &[Vec<usize>],
    numers: &[AB::Expr],
    denoms: &[AB::ExprEF],
    local: &[AB::VarEF],
    next: &[AB::VarEF],
    sum: AB::ExprEF,
) {
    let chunk_local = &local[..interaction_chunks.len()];
    let chunk_next = &next[..interaction_chunks.len()];
    let sum_local = local[interaction_chunks.len()];
    let sum_next = next[interaction_chunks.len()];

    izip!(chunk_local, interaction_chunks).for_each(|(chunk_sum, chunk)| {
        let lhs = chunk
            .iter()
            .fold((*chunk_sum).into(), |acc, i| acc * denoms[*i].clone());
        let rhs = if chunk.len() == 1 {
            numers[chunk[0]].clone().into()
        } else {
            chunk
                .iter()
                .map(|i| {
                    chunk
                        .iter()
                        .filter(|j| i != *j)
                        .map(|j| denoms[*j].clone())
                        .product::<AB::ExprEF>()
                        * numers[*i].clone()
                })
                .sum::<AB::ExprEF>()
        };

        builder.assert_eq_ext(lhs, rhs);
    });

    builder.when_transition().assert_eq_ext(
        sum_next.into() - sum_local.into(),
        chunk_next.iter().copied().map_into().sum::<AB::ExprEF>(),
    );
    builder.when_first_row().assert_eq_ext(
        sum_local,
        chunk_local.iter().copied().map_into().sum::<AB::ExprEF>(),
    );
    builder.when_last_row().assert_eq_ext(sum_local, sum);
}
