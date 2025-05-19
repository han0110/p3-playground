# Plonky3 Playground

Experimental crates based on [Plonky3](https://github.com/Plonky3/Plonky3).

## Acknowledgements

- `p3-air-ext` includes:
  - Implementations `DebugConstraintBuilder` and `SymbolicAirBuilder` originally from `p3-uni-stark` of [Plonky3](https://github.com/Plonky3/Plonky3).
  - Traits `InteractionBuilder` and `SubAirBuilder` originally designed by [SP1](https://github.com/succinctlabs/sp1).
- `p3-fri-ext` is duplicated from `p3-fri` of [Plonky3](https://github.com/Plonky3/Plonky3), and with the modification of FRI higher arity implemented by [Axiom](https://github.com/Plonky3/Plonky3/pull/592).
- `p3-uni-stark-ext` is duplicated from `p3-uni-stark` of [Plonky3](https://github.com/Plonky3/Plonky3), and modified to support `InteractionBuilder`.
- `p3-hyperplonk` is inspired by [Whirlaway](https://github.com/TomWambsgans/Whirlaway) and [Binius](https://github.com/IrreducibleOSS/binius) to use [Gruen's Univariate Skip](https://eprint.iacr.org/2024/108) to speed up the first few rounds of Sumcheck.
- `p3-whir` wraps [whir-p3](https://github.com/tcoratger/whir-p3) to fit the multilinear PCS trait.
