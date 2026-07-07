//! `calibrate` — placeholder. v1 ships no calibration: thresholds stay near `i16::MAX`
//! (silent above L0). A later phase will lower per-layer thresholds toward a firing-rate
//! target and set each layer's saturation to a margin above its threshold band,
//! maintaining `saturation >= max threshold`.
