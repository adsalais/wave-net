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
const VERSION: u16 = 1;

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

#[allow(dead_code)] // used by the runtime format (Task 4)
fn w_vec_u8(w: &mut impl Write, v: &[u8]) -> io::Result<()> { w_u64(w, v.len() as u64)?; w.write_all(v) }
#[allow(dead_code)]
fn w_vec_i32(w: &mut impl Write, v: &[i32]) -> io::Result<()> { w_u64(w, v.len() as u64)?; for &x in v { w_i32(w, x)?; } Ok(()) }
#[allow(dead_code)]
fn r_vec_u8(r: &mut impl Read) -> io::Result<Vec<u8>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_u8(r)?); } Ok(v) }
#[allow(dead_code)]
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
        w_u16(&mut buf, VERSION)?;
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
        if ver != VERSION {
            return Err(inval(format!("unsupported model version {ver} (expected {VERSION})")));
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
                    for s in 0..lb.shadow.len() {
                        assert_eq!(lb.shadow[s], lb.weight_at(s) as f32);
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
        w_u16(&mut bad, VERSION).unwrap();
        let ck = fnv1a(&bad);
        w_u64(&mut bad, ck).unwrap();
        assert_eq!(err_kind(Network::load_model(&bad[..])), io::ErrorKind::InvalidData);

        // unsupported version
        let mut badv = MAGIC_MODEL.to_vec();
        w_u16(&mut badv, VERSION + 1).unwrap();
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
    fn model_path_roundtrip() {
        let net = small_net();
        let path = std::env::temp_dir().join("wave_bitnet_persist_model_test.wbm");
        net.save_model_path(&path).unwrap();
        let loaded = Network::load_model_path(&path).unwrap();
        assert_models_eq(&net, &loaded);
        let _ = std::fs::remove_file(&path);
    }
}
