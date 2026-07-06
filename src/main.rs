//! Smoke entry: build the demo reservoir, run a few waves, report activity. Proves the engine
//! base compiles and runs (`cargo run`). The learning RSNN will be built on `wave_net`'s lib.

use wave_net::wave_reservoir::config::IntConfig;
use wave_net::wave_reservoir::pipeline::LayerNet;

fn main() {
    let cfg = IntConfig::demo();
    let n = cfg.n_total();
    let ls = (cfg.w * cfg.h) as usize;
    let mut drive = vec![0i16; n];
    for d in drive.iter_mut().take(ls) {
        *d = 50; // sustained bottom-layer input
    }
    let net = LayerNet::new(cfg);
    for _ in 0..10 {
        net.wave(&drive);
    }
    let active = (0..n).filter(|&i| net.potential_global(i) != 0).count();
    println!("wave-net: LayerNet ran 10 waves on the demo reservoir; {active}/{n} neurons non-zero.");
}
