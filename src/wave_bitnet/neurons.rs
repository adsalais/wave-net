//! `neurons` — a `Layer`'s per-neuron state and its bitset weight representation. The topology is a
//! per-neuron neighborhood occupancy bitset; weights are 2-bit (nonzero + sign) with an f32 training
//! shadow. (Const definitions here so `config::validate` can reference `ADAPT_SHIFT`; the `Layer`
//! struct + impl are added in the same file.)

/// Fixed-point scale for `adapt`: it holds the effective threshold contribution × `2^ADAPT_SHIFT`.
/// Bounded by the i32 overflow limit on the bump-add (`2·ADAPT_MAX = i16::MAX << (SHIFT+1)` must fit
/// i32, i.e. `SHIFT <= 14`); 12 keeps ~8× margin and allows `adapt_decay` up to 12 (τ ≈ 4096 waves).
pub const ADAPT_SHIFT: u32 = 12;
/// Ceiling for `adapt`, so the effective contribution never exceeds `i16::MAX` (overflow guard).
pub const ADAPT_MAX: i32 = (i16::MAX as i32) << ADAPT_SHIFT;
