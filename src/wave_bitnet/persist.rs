//! `persist` — hand-rolled, std-only binary save/load for a `wave_bitnet` `Network`. Two independent
//! formats: a self-contained **model** (`b"WBNM"`: structure + 2-bit codes, inference-ready) and a
//! **runtime overlay** (`b"WBNR"`: the mutable per-neuron state only) applied onto a loaded model.
//! See docs/superpowers/specs/2026-07-13-wave-bitnet-persist-design.md.

use crate::wave_bitnet::network::Network;
use crate::wave_bitnet::neurons::Layer;
use crate::wave_bitnet::synapse::TopologyLevel;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC_MODEL: &[u8; 4] = b"WBNM";
const MAGIC_RUNTIME: &[u8; 4] = b"WBNR";
/// Model file format version (structure + codes). Unchanged by the optional-`TrainState` work.
const MODEL_VERSION: u16 = 1;
/// Runtime overlay format version. Bumped to 2 when the overlay was slimmed to the genuinely
/// resumable forward state (`potential/cooldown/adapt/pending` + `wave_id`); older overlays rejected.
const RUNTIME_VERSION: u16 = 2;

fn inval(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// ---- FNV-1a 64-bit over raw bytes (file integrity + model fingerprint) ----
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---- byte primitives (little-endian). Some are first used by the runtime format (Task 4). ----
fn w_u8(w: &mut impl Write, v: u8) -> io::Result<()> { w.write_all(&[v]) }
fn w_u16(w: &mut impl Write, v: u16) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_u32(w: &mut impl Write, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_u64(w: &mut impl Write, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_i16(w: &mut impl Write, v: i16) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_i32(w: &mut impl Write, v: i32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_f32(w: &mut impl Write, v: f32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }

fn r_u8(r: &mut impl Read) -> io::Result<u8> { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0]) }
fn r_u16(r: &mut impl Read) -> io::Result<u16> { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn r_u32(r: &mut impl Read) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn r_u64(r: &mut impl Read) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn r_i16(r: &mut impl Read) -> io::Result<i16> { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(i16::from_le_bytes(b)) }
fn r_i32(r: &mut impl Read) -> io::Result<i32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) }
fn r_f32(r: &mut impl Read) -> io::Result<f32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }

// length-prefixed vectors (u64 length; no pre-allocation, so a corrupt length fails fast on read).
fn w_vec_i16(w: &mut impl Write, v: &[i16]) -> io::Result<()> { w_u64(w, v.len() as u64)?; for &x in v { w_i16(w, x)?; } Ok(()) }
fn w_vec_u64(w: &mut impl Write, v: &[u64]) -> io::Result<()> { w_u64(w, v.len() as u64)?; for &x in v { w_u64(w, x)?; } Ok(()) }
fn r_vec_i16(r: &mut impl Read) -> io::Result<Vec<i16>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_i16(r)?); } Ok(v) }
fn r_vec_u64(r: &mut impl Read) -> io::Result<Vec<u64>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_u64(r)?); } Ok(v) }

fn w_vec_u8(w: &mut impl Write, v: &[u8]) -> io::Result<()> { w_u64(w, v.len() as u64)?; w.write_all(v) }
fn w_vec_i32(w: &mut impl Write, v: &[i32]) -> io::Result<()> { w_u64(w, v.len() as u64)?; for &x in v { w_i32(w, x)?; } Ok(()) }
fn r_vec_u8(r: &mut impl Read) -> io::Result<Vec<u8>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_u8(r)?); } Ok(v) }
fn r_vec_i32(r: &mut impl Read) -> io::Result<Vec<i32>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_i32(r)?); } Ok(v) }

/// Serialize the model IDENTITY: dims + per-layer structure + 2-bit codes. No magic/version/checksum,
/// no runtime state — this is exactly what `model_fingerprint` (Task 4) hashes and what the model file
/// carries between its header and trailing checksum.
fn write_model_identity(w: &mut impl Write, net: &Network) -> io::Result<()> {
    w_u32(w, net.size())?;
    w_u32(w, net.layer_count() as u32)?;
    for lz in net.layers() {
        w_u32(w, lz.topology.len() as u32)?;
        for t in &lz.topology {
            w_i32(w, t.level)?;
            w_u32(w, t.radius)?;
            w_u32(w, t.count)?;
        }
        w_u8(w, lz.leak.0)?;
        w_u8(w, lz.leak.1)?;
        w_u8(w, lz.cooldown_base)?;
        w_i16(w, lz.adapt_bump)?;
        w_u8(w, lz.adapt_decay)?;
        w_u8(w, lz.readout as u8)?;
        w_f32(w, lz.ternary_threshold)?;
        w_vec_i16(w, &lz.threshold)?;
        for level in &lz.occ {
            w_vec_u64(w, level)?;
        }
        w_vec_u64(w, &lz.codes)?;
    }
    Ok(())
}

/// Read a model identity from `r` (already past magic/version) and assemble a `Network`.
fn read_model_body(r: &mut impl Read) -> io::Result<Network> {
    let size = r_u32(r)?;
    let n_layers = r_u32(r)? as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let n_lv = r_u32(r)? as usize;
        let mut topology = Vec::with_capacity(n_lv);
        for _ in 0..n_lv {
            topology.push(TopologyLevel { level: r_i32(r)?, radius: r_u32(r)?, count: r_u32(r)? });
        }
        let leak = (r_u8(r)?, r_u8(r)?);
        let cooldown_base = r_u8(r)?;
        let adapt_bump = r_i16(r)?;
        let adapt_decay = r_u8(r)?;
        let readout = r_u8(r)? != 0;
        let ternary_threshold = r_f32(r)?;
        let threshold = r_vec_i16(r)?;
        let mut occ = Vec::with_capacity(n_lv);
        for _ in 0..n_lv {
            occ.push(r_vec_u64(r)?);
        }
        let codes = r_vec_u64(r)?;
        let layer = Layer::from_parts(
            topology, leak, cooldown_base, adapt_bump, adapt_decay, readout, ternary_threshold,
            threshold, occ, codes, size,
        )
        .map_err(inval)?;
        layers.push(layer);
    }
    Ok(Network::from_layers(size, layers))
}

impl Network {
    /// Serialize a self-contained, inference-ready model (`b"WBNM"`): materialized structure
    /// (occupancy + thresholds) + 2-bit weight codes. No seed dependence, no f32 shadow.
    pub fn save_model(&self, mut w: impl Write) -> io::Result<()> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_MODEL)?;
        w_u16(&mut buf, MODEL_VERSION)?;
        write_model_identity(&mut buf, self)?;
        let cksum = fnv1a(&buf);
        w_u64(&mut buf, cksum)?;
        w.write_all(&buf)
    }

    /// Load a model saved by `save_model`. Verifies the trailing checksum, magic, and version;
    /// reconstructs each layer (shadow = decode of codes; runtime zeroed). Fails loud on any mismatch.
    pub fn load_model(mut r: impl Read) -> io::Result<Network> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        if buf.len() < MAGIC_MODEL.len() + 2 + 8 {
            return Err(inval("model file too short"));
        }
        let (body, cks) = buf.split_at(buf.len() - 8);
        let stored = u64::from_le_bytes(cks.try_into().unwrap());
        if fnv1a(body) != stored {
            return Err(inval("model checksum mismatch (corrupt file)"));
        }
        let mut c = io::Cursor::new(body);
        let mut magic = [0u8; 4];
        c.read_exact(&mut magic)?;
        if &magic != MAGIC_MODEL {
            return Err(inval("bad magic (not a wave_bitnet model file)"));
        }
        let ver = r_u16(&mut c)?;
        if ver != MODEL_VERSION {
            return Err(inval(format!("unsupported model version {ver} (expected {MODEL_VERSION})")));
        }
        read_model_body(&mut c)
    }

    /// Convenience: `save_model` to a file path.
    pub fn save_model_path(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_model(File::create(path)?)
    }

    /// Convenience: `load_model` from a file path.
    pub fn load_model_path(path: impl AsRef<Path>) -> io::Result<Network> {
        Network::load_model(File::open(path)?)
    }

    /// Stable 64-bit fingerprint of a model's identity (structure + codes), independent of runtime
    /// state. Written into a runtime overlay and re-checked on `apply_runtime` so an overlay can only
    /// be applied to the model it was taken from.
    pub fn model_fingerprint(&self) -> u64 {
        let mut buf = Vec::new();
        write_model_identity(&mut buf, self).expect("Vec<u8> write is infallible");
        fnv1a(&buf)
    }

    /// Serialize ONLY the genuinely-resumable per-neuron forward state + `wave_id` (`b"WBNR"`):
    /// `potential`, `cooldown`, `adapt`, `pending`. Not standalone: `apply_runtime` restores it onto a
    /// matching model. `scratch.deliv` is provably all-zero between waves, so it is not stored; training
    /// state (shadow / decide snapshots) lives in `Layer.train` and is never part of the overlay.
    pub fn save_runtime(&self, mut w: impl Write) -> io::Result<()> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_RUNTIME)?;
        w_u16(&mut buf, RUNTIME_VERSION)?;
        w_u64(&mut buf, self.model_fingerprint())?;
        w_u32(&mut buf, self.size())?;
        w_u32(&mut buf, self.layer_count() as u32)?;
        w_u64(&mut buf, self.wave_id() as u64)?;
        for lz in self.layers() {
            w_vec_i16(&mut buf, &lz.potential)?;
            w_vec_u8(&mut buf, &lz.cooldown)?;
            w_vec_i32(&mut buf, &lz.adapt)?;
            w_vec_i32(&mut buf, &lz.pending)?;
        }
        let cksum = fnv1a(&buf);
        w_u64(&mut buf, cksum)?;
        w.write_all(&buf)
    }

    /// Restore a runtime overlay (from `save_runtime`) onto this network, in place. Verifies checksum,
    /// magic, version, that the overlay's `model_fingerprint` matches this model, and that every array
    /// length matches, all BEFORE mutating any state. Fails loud on any mismatch.
    pub fn apply_runtime(&mut self, mut r: impl Read) -> io::Result<()> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        if buf.len() < MAGIC_RUNTIME.len() + 2 + 8 {
            return Err(inval("runtime file too short"));
        }
        let (body, cks) = buf.split_at(buf.len() - 8);
        let stored = u64::from_le_bytes(cks.try_into().unwrap());
        if fnv1a(body) != stored {
            return Err(inval("runtime checksum mismatch (corrupt file)"));
        }
        let mut c = io::Cursor::new(body);
        let mut magic = [0u8; 4];
        c.read_exact(&mut magic)?;
        if &magic != MAGIC_RUNTIME {
            return Err(inval("bad magic (not a wave_bitnet runtime file)"));
        }
        let ver = r_u16(&mut c)?;
        if ver != RUNTIME_VERSION {
            return Err(inval(format!("unsupported runtime version {ver} (expected {RUNTIME_VERSION})")));
        }
        let model_fp = r_u64(&mut c)?;
        if model_fp != self.model_fingerprint() {
            return Err(inval("runtime overlay does not belong to this model (fingerprint mismatch)"));
        }
        let size = r_u32(&mut c)?;
        let n_layers = r_u32(&mut c)? as usize;
        if size != self.size() || n_layers != self.layer_count() {
            return Err(inval("runtime dims do not match this model"));
        }
        let wave_id = r_u64(&mut c)? as usize;
        let ls = (size as usize) * (size as usize);
        // read + validate every layer's arrays BEFORE mutating (keep apply all-or-nothing)
        type LayerRt = (Vec<i16>, Vec<u8>, Vec<i32>, Vec<i32>);
        let mut staged: Vec<LayerRt> = Vec::with_capacity(n_layers);
        for z in 0..n_layers {
            let potential = r_vec_i16(&mut c)?;
            let cooldown = r_vec_u8(&mut c)?;
            let adapt = r_vec_i32(&mut c)?;
            let pending = r_vec_i32(&mut c)?;
            for (name, len) in [
                ("potential", potential.len()), ("cooldown", cooldown.len()),
                ("adapt", adapt.len()), ("pending", pending.len()),
            ] {
                if len != ls {
                    return Err(inval(format!("layer {z} {name} length {len} != ls {ls}")));
                }
            }
            staged.push((potential, cooldown, adapt, pending));
        }
        // commit
        let layers = self.layers_mut();
        for (z, (potential, cooldown, adapt, pending)) in staged.into_iter().enumerate() {
            layers[z].potential = potential;
            layers[z].cooldown = cooldown;
            layers[z].adapt = adapt;
            layers[z].pending = pending;
        }
        self.set_wave_id(wave_id);
        Ok(())
    }

    /// Convenience: `save_runtime` to a file path.
    pub fn save_runtime_path(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_runtime(File::create(path)?)
    }

    /// Convenience: `apply_runtime` from a file path.
    pub fn apply_runtime_path(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.apply_runtime(File::open(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::{Config, LayerConfig};
    use crate::wave_bitnet::synapse::TopologyLevel;

    fn small_net() -> Network {
        let up = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 9830, threshold_jitter: 64,
            baseline_init: 6, adapt_bump: 5, adapt_decay: 6,
        };
        let mid = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 8 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
            ],
            ..up.clone()
        };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Network::new_with_readout(Config { seed: 0xABCD, size: 8, layers: vec![up, mid, top] })
    }

    /// `ErrorKind` of a failed load (avoids `unwrap_err`, which would need `Network: Debug`).
    fn err_kind(r: io::Result<Network>) -> io::ErrorKind {
        match r {
            Ok(_) => panic!("expected an error, got Ok"),
            Err(e) => e.kind(),
        }
    }

    /// `ErrorKind` of a failed `apply_runtime` (which returns `io::Result<()>`).
    fn err_kind_rt(r: io::Result<()>) -> io::ErrorKind {
        match r {
            Ok(()) => panic!("expected an error, got Ok"),
            Err(e) => e.kind(),
        }
    }

    #[test]
    fn runtime_version_is_bumped_and_distinct_from_model() {
        assert_eq!(MODEL_VERSION, 1, "model format unchanged");
        assert_eq!(RUNTIME_VERSION, 2, "runtime overlay bumped for the slimmer layout");
    }

    #[test]
    fn runtime_rejects_old_version() {
        // hand-build a runtime header with the OLD version byte (1); expect a loud reject.
        let net = small_net();
        let mut bad = MAGIC_RUNTIME.to_vec();
        w_u16(&mut bad, 1).unwrap(); // stale runtime version
        w_u64(&mut bad, net.model_fingerprint()).unwrap();
        w_u32(&mut bad, net.size()).unwrap();
        w_u32(&mut bad, net.layer_count() as u32).unwrap();
        w_u64(&mut bad, 0).unwrap();
        let ck = fnv1a(&bad);
        w_u64(&mut bad, ck).unwrap();
        let mut n = small_net();
        assert_eq!(err_kind_rt(n.apply_runtime(&bad[..])), io::ErrorKind::InvalidData);
    }

    /// Assert two networks have equal persisted + derived structure (not runtime; shadow checked as
    /// the decode of codes).
    fn assert_models_eq(a: &Network, b: &Network) {
        assert_eq!(a.size(), b.size());
        assert_eq!(a.layer_count(), b.layer_count());
        for z in 0..a.layer_count() {
            a.with_layer(z, |la| {
                b.with_layer(z, |lb| {
                    assert_eq!(la.topology, lb.topology, "layer {z} topology");
                    assert_eq!(la.leak, lb.leak);
                    assert_eq!(la.cooldown_base, lb.cooldown_base);
                    assert_eq!(la.adapt_bump, lb.adapt_bump);
                    assert_eq!(la.adapt_decay, lb.adapt_decay);
                    assert_eq!(la.readout, lb.readout);
                    assert_eq!(la.ternary_threshold, lb.ternary_threshold);
                    assert_eq!(la.threshold, lb.threshold, "layer {z} threshold");
                    assert_eq!(la.occ, lb.occ, "layer {z} occ");
                    assert_eq!(la.codes, lb.codes, "layer {z} codes");
                    assert_eq!(la.total_slots, lb.total_slots);
                    assert_eq!(la.slot_bases, lb.slot_bases);
                    assert_eq!(la.neigh, lb.neigh);
                    assert_eq!(la.occ_wpn, lb.occ_wpn);
                    assert_eq!(la.offsets, lb.offsets);
                    assert_eq!(la.off_flat, lb.off_flat);
                    // loaded shadow is the decode of codes
                    let lb_shadow = &lb.train.as_ref().unwrap().shadow;
                    for s in 0..lb_shadow.len() {
                        assert_eq!(lb_shadow[s], lb.weight_at(s) as f32);
                    }
                })
            });
        }
    }

    #[test]
    fn model_roundtrip_fields() {
        let net = small_net();
        let mut bytes = Vec::new();
        net.save_model(&mut bytes).unwrap();
        let loaded = Network::load_model(&bytes[..]).unwrap();
        assert_models_eq(&net, &loaded);
    }

    #[test]
    fn model_roundtrip_inference_equivalent() {
        let net = small_net();
        let mut bytes = Vec::new();
        net.save_model(&mut bytes).unwrap();
        let mut a = Network::load_model(&bytes[..]).unwrap();
        let mut b = Network::load_model(&bytes[..]).unwrap();
        let inputs: [&[u32]; 6] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[], &[]];
        for inp in inputs {
            a.wave(inp);
            b.wave(inp);
            for z in 0..a.layer_count() {
                assert_eq!(a.layer_decide_potential(z), b.layer_decide_potential(z), "layer {z} decide_potential");
            }
        }
    }

    #[test]
    fn model_bytes_are_stable() {
        let net = small_net();
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        net.save_model(&mut b1).unwrap();
        net.save_model(&mut b2).unwrap();
        assert_eq!(b1, b2);
    }

    #[test]
    fn model_rejects_bad_magic_version_and_corruption() {
        // bad magic (valid checksum so magic check is reached)
        let mut bad = b"ZZZZ".to_vec();
        w_u16(&mut bad, MODEL_VERSION).unwrap();
        let ck = fnv1a(&bad);
        w_u64(&mut bad, ck).unwrap();
        assert_eq!(err_kind(Network::load_model(&bad[..])), io::ErrorKind::InvalidData);

        // unsupported version
        let mut badv = MAGIC_MODEL.to_vec();
        w_u16(&mut badv, MODEL_VERSION + 1).unwrap();
        let ck = fnv1a(&badv);
        w_u64(&mut badv, ck).unwrap();
        assert_eq!(err_kind(Network::load_model(&badv[..])), io::ErrorKind::InvalidData);

        // corrupt body (flip a payload byte)
        let net = small_net();
        let mut good = Vec::new();
        net.save_model(&mut good).unwrap();
        good[8] ^= 0xFF;
        assert_eq!(err_kind(Network::load_model(&good[..])), io::ErrorKind::InvalidData);

        // truncated
        let net = small_net();
        let mut full = Vec::new();
        net.save_model(&mut full).unwrap();
        let half = &full[..full.len() / 2];
        assert!(Network::load_model(half).is_err());
    }

    #[test]
    fn runtime_resume_equivalent() {
        let net = small_net();
        let mut model_bytes = Vec::new();
        net.save_model(&mut model_bytes).unwrap();

        // A: load, drive several waves, snapshot runtime
        let mut a = Network::load_model(&model_bytes[..]).unwrap();
        let inputs: [&[u32]; 5] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[]];
        for inp in inputs {
            a.wave(inp);
        }
        let mut rt_bytes = Vec::new();
        a.save_runtime(&mut rt_bytes).unwrap();

        // B: fresh load of the SAME model, apply A's runtime overlay
        let mut b = Network::load_model(&model_bytes[..]).unwrap();
        b.apply_runtime(&rt_bytes[..]).unwrap();

        // immediate state equality
        assert_eq!(a.wave_id(), b.wave_id());
        for z in 0..a.layer_count() {
            a.with_layer(z, |la| {
                b.with_layer(z, |lb| {
                    assert_eq!(la.potential, lb.potential, "layer {z} potential");
                    assert_eq!(la.cooldown, lb.cooldown, "layer {z} cooldown");
                    assert_eq!(la.adapt, lb.adapt, "layer {z} adapt");
                    assert_eq!(la.pending, lb.pending, "layer {z} pending");
                })
            });
        }

        // continuing both must stay identical (true resume)
        let more: [&[u32]; 3] = [&[1, 4], &[], &[7]];
        for inp in more {
            a.wave(inp);
            b.wave(inp);
            for z in 0..a.layer_count() {
                assert_eq!(a.layer_decide_potential(z), b.layer_decide_potential(z), "resumed layer {z}");
            }
        }
    }

    #[test]
    fn runtime_binding_rejects_wrong_model() {
        // source model + its runtime
        let a = small_net();
        let mut model_a = Vec::new();
        a.save_model(&mut model_a).unwrap();
        let mut a = Network::load_model(&model_a[..]).unwrap();
        a.wave(&[0, 1, 2]);
        let mut rt = Vec::new();
        a.save_runtime(&mut rt).unwrap();

        // a DIFFERENT model (different seed → different occupancy/codes → different fingerprint)
        let other = {
            let up = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }],
                leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 9830, threshold_jitter: 64,
                baseline_init: 6, adapt_bump: 5, adapt_decay: 6,
            };
            let mid = LayerConfig {
                topology: vec![
                    TopologyLevel { level: 1, radius: 2, count: 8 },
                    TopologyLevel { level: 0, radius: 1, count: 2 },
                ],
                ..up.clone()
            };
            let top = LayerConfig { topology: vec![], ..up.clone() };
            Network::new_with_readout(Config { seed: 0x9999, size: 8, layers: vec![up, mid, top] })
        };
        let mut other_bytes = Vec::new();
        other.save_model(&mut other_bytes).unwrap();
        let mut other = Network::load_model(&other_bytes[..]).unwrap();

        match other.apply_runtime(&rt[..]) {
            Ok(()) => panic!("expected a fingerprint-mismatch error"),
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
        }
    }

    #[test]
    fn runtime_fingerprint_stable_across_save_load() {
        let net = small_net();
        let fp = net.model_fingerprint();
        let mut bytes = Vec::new();
        net.save_model(&mut bytes).unwrap();
        let loaded = Network::load_model(&bytes[..]).unwrap();
        assert_eq!(fp, loaded.model_fingerprint());
    }

    #[test]
    fn runtime_path_roundtrip() {
        let net = small_net();
        let mut model_bytes = Vec::new();
        net.save_model(&mut model_bytes).unwrap();
        let mut a = Network::load_model(&model_bytes[..]).unwrap();
        a.wave(&[0, 1, 2]);
        a.wave(&[]);

        let path = std::env::temp_dir().join("wave_bitnet_persist_runtime_test.wbr");
        a.save_runtime_path(&path).unwrap();
        let mut b = Network::load_model(&model_bytes[..]).unwrap();
        b.apply_runtime_path(&path).unwrap();
        assert_eq!(a.wave_id(), b.wave_id());
        a.with_layer(1, |la| b.with_layer(1, |lb| assert_eq!(la.potential, lb.potential)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn model_path_roundtrip() {
        let net = small_net();
        let path = std::env::temp_dir().join("wave_bitnet_persist_model_test.wbm");
        net.save_model_path(&path).unwrap();
        let loaded = Network::load_model_path(&path).unwrap();
        assert_models_eq(&net, &loaded);
        let _ = std::fs::remove_file(&path);
    }
}
