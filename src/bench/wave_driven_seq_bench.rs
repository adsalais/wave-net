//! Sequence-recall memory benchmark for the `wave_driven` engine (test-only). Asks whether the engine
//! can *memorize* a branching sequence set — reproduce deterministic continuations, match fork
//! marginals as calibrated readout mass, and resolve a prefix family that only a 3-token memory can
//! answer. Self-contained (its own 9-class readout; `wave_driven_bench`'s 2-class path is untouched).
//! Spec: docs/superpowers/specs/2026-07-15-wave-driven-sequence-memory-design.md

#[cfg(test)]
mod tests {
    use crate::wave_driven::synapse::{key, mix};

    /// Vocabulary size. Tokens {1,2,3,4,5,6,7,8,16} → ids 0..8.
    const V: usize = 9;
    /// Sequences are 4 tokens: a prefix of 1..=3, then the target.
    const MAX_PREFIX: usize = 3;
    /// Hash purpose tag for trial sampling (distinct stream from CUE_P / P_DFA).
    const P_SEQ: u64 = 0x5E9;
    /// Hash purpose tag for token site codes. Distinct from `wave_driven_bench`'s `CUE_P` (0xC0E) —
    /// this is a different predicate over a different site set, and a different task.
    const CUE_P: u64 = 0xC0F;

    /// The six sequences as token ids. Sets are nested: set 4 = SEQS[..4], set 5 = SEQS[..5], etc.
    /// ids: 0→"1" 1→"2" 2→"3" 3→"4" 4→"5" 5→"6" 6→"7" 7→"8" 8→"16"
    ///
    /// S5 and S6 deliberately extend the same `2→3` collision S4 introduced, so growing the set
    /// deepens the memory test rather than only adding capacity: the `[·,2,3]` family goes 2→3→4-way
    /// and the Markov-2 ceiling falls 50%→33%→25% while true memory stays at 100%.
    const SEQS: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // 1→2→3→4
        [0, 1, 3, 7], // 1→2→4→8
        [0, 3, 7, 8], // 1→4→8→16
        [1, 1, 2, 4], // 2→2→3→5
        [2, 1, 2, 5], // 3→2→3→6
        [3, 1, 2, 6], // 4→2→3→7
    ];

    /// Trial generator: sequence uniform over the set, prefix length uniform in 1..=MAX_PREFIX,
    /// target = the next token. Deterministic in `trial`; matches the harness convention
    /// `Fn(task_seed, trial) -> (prefix, target)`.
    ///
    /// Uniform sequence sampling is what produces the target conditionals for free: conditioned on
    /// prefix `[1]`, the sequence is uniform over {S1,S2,S3}, giving {2: 2/3, 4: 1/3}.
    fn seq_task(set_size: usize) -> impl Fn(u64, usize) -> (Vec<usize>, usize) {
        move |task_seed, trial| {
            let s = (mix(key(task_seed, trial as u32, 0, 0, P_SEQ)) % set_size as u64) as usize;
            let n = (mix(key(task_seed, trial as u32, 0, 1, P_SEQ)) % MAX_PREFIX as u64) as usize + 1;
            (SEQS[s][..n].to_vec(), SEQS[s][n])
        }
    }

    /// Every distinct prefix of length 1..=MAX_PREFIX in the set, in deterministic discovery order.
    fn prefixes(set_size: usize) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::new();
        for s in 0..set_size {
            for n in 1..=MAX_PREFIX {
                let p = SEQS[s][..n].to_vec();
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Closed-form P(next | prefix) over the V tokens, under uniform sampling of the set.
    fn conditional(set_size: usize, prefix: &[usize]) -> Vec<f32> {
        let mut counts = vec![0f32; V];
        let mut total = 0f32;
        for s in 0..set_size {
            if SEQS[s][..prefix.len()] == *prefix {
                counts[SEQS[s][prefix.len()]] += 1.0;
                total += 1.0;
            }
        }
        counts.iter().map(|c| c / total).collect()
    }

    /// The `[·,2,3]` disambiguation family: the length-3 prefixes sharing the (2,3) suffix and
    /// differing only in the first token. A Markov-2 model cannot separate these; only a 3-token
    /// memory can. Token ids: "2" = 1, "3" = 2.
    fn family(set_size: usize) -> Vec<Vec<usize>> {
        prefixes(set_size).into_iter().filter(|p| p.len() == MAX_PREFIX && p[1] == 1 && p[2] == 2).collect()
    }

    /// The last `min(k, len)` tokens of a prefix — a Markov-k model's context.
    fn ctx_of(p: &[usize], k: usize) -> Vec<usize> {
        let n = p.len().min(k);
        p[p.len() - n..].to_vec()
    }

    /// Sampling weight of a prefix: proportional to the number of sequences carrying it (the
    /// uniform-over-prefix-length factor is constant and cancels).
    fn prefix_weight(set_size: usize, p: &[usize]) -> f32 {
        (0..set_size).filter(|&s| SEQS[s][..p.len()] == *p).count() as f32
    }

    /// Expected accuracy of a Markov-k model on `targets`, under the model's own predictive
    /// distribution (so ties need no tie-breaking rule: a model spreading mass over k options scores
    /// exactly 1/k). The model is fit in closed form from the set: group every prefix by its
    /// length-k context, then average their conditionals weighted by sampling frequency.
    fn markov_k_accuracy(set_size: usize, k: usize, targets: &[Vec<usize>]) -> f32 {
        let all = prefixes(set_size);
        let mut acc = 0f32;
        for p in targets {
            let ctx = ctx_of(p, k);
            let mut counts = vec![0f32; V];
            let mut total = 0f32;
            for q in &all {
                if ctx_of(q, k) == ctx {
                    let w = prefix_weight(set_size, q);
                    let cond = conditional(set_size, q);
                    for t in 0..V {
                        counts[t] += w * cond[t];
                    }
                    total += w;
                }
            }
            let qdist: Vec<f32> = counts.iter().map(|c| c / total).collect();
            let truth = conditional(set_size, p);
            acc += (0..V).map(|t| truth[t] * qdist[t]).sum::<f32>();
        }
        acc / targets.len() as f32
    }

    /// A token's L0 site code: a fixed random subset of the grid, `density`/8 of all sites
    /// (density 1 ≈ 32 sites, density 2 ≈ 64 of 256).
    ///
    /// Population-coded, not place-coded, for two reasons. (1) Arithmetic: the engine's leak floor
    /// (`wave.rs`, `d.max(1)`) drains ≥1 per wave, so a single +1 synapse nets zero and a lone site
    /// can never fire anything — ≥2 coincident synapses is a precondition for any activity, and
    /// `sample_distinct_cells` caps a source at one synapse per target. (2) Science: random codes
    /// share no exploitable structure, so the net cannot interpolate geometrically instead of
    /// remembering. See the spec's Design analysis §1-2.
    fn token_sites(task_seed: u64, size: u32, token: usize, density: u32) -> Vec<u32> {
        let ls = size * size;
        (0..ls).filter(|&loc| (mix(key(task_seed, loc, token as i32, 0, CUE_P)) & 7) < density as u64).collect()
    }

    /// Max-subtract softmax over V logits.
    fn softmax_n(z: &[f32]) -> Vec<f32> {
        let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = z.iter().map(|x| (x - m).exp()).collect();
        let s = e.iter().sum::<f32>().max(1e-30);
        e.iter().map(|x| x / s).collect()
    }

    /// Per-class score: the dot product of each class's readout weights with the spike counts.
    fn score_n(w: &[Vec<f32>], a: &[f32]) -> Vec<f32> {
        w.iter().map(|wc| wc.iter().zip(a).map(|(x, y)| x * y).sum()).collect()
    }

    /// Argmax breaking ties toward the lower index — matching the 2-class harness's `(s1 > s0)`
    /// rule, which predicts class 0 on a tie. At init every weight is 0 and every score ties at 0.0,
    /// so the tie-break is reachable, not theoretical.
    fn argmax_first(z: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, &x) in z.iter().enumerate() {
            if x > z[best] {
                best = i;
            }
        }
        best
    }

    /// Total variation distance, ½·Σ|p−q|. Bounded and legible: 0 is perfect, 0.5 means a 50/50 fork
    /// collapsed onto one branch. For a one-hot `p`, `1 − TV` is exactly `q[target]`.
    fn total_variation(p: &[f32], q: &[f32]) -> f32 {
        0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f32>()
    }

    #[test]
    fn token_sites_density_and_determinism() {
        // Density 1/8 ≈ 32 sites, 2/8 ≈ 64 sites of 256 (binomial, so a generous band).
        for token in 0..V {
            let n32 = token_sites(11, 16, token, 1).len();
            let n64 = token_sites(11, 16, token, 2).len();
            assert!((18..=50).contains(&n32), "density 1 ≈ 32 sites, got {n32} for token {token}");
            assert!((44..=86).contains(&n64), "density 2 ≈ 64 sites, got {n64} for token {token}");
        }

        // Determinism: a pure function of its arguments.
        assert_eq!(token_sites(11, 16, 3, 2), token_sites(11, 16, 3, 2));

        // Distinct tokens get distinct codes (random population codes overlap, but not wholly).
        let a = token_sites(11, 16, 0, 2);
        let b = token_sites(11, 16, 1, 2);
        assert_ne!(a, b, "distinct tokens must not share a code");
        let shared = a.iter().filter(|s| b.contains(s)).count();
        assert!(shared < a.len() * 3 / 4, "token codes must stay separable, shared {shared} of {}", a.len());

        // Sites are in range and strictly ascending (the filter preserves order).
        let s = token_sites(11, 16, 5, 2);
        assert!(s.windows(2).all(|w| w[0] < w[1]), "sites ascend");
        assert!(s.iter().all(|&loc| loc < 256), "sites are in range");
    }

    #[test]
    fn readout_primitives_correct() {
        // softmax_n is a distribution, and max-subtraction keeps it finite on large inputs.
        let p = softmax_n(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0], "monotone in the logit");
        let big = softmax_n(&[1000.0, 1000.0]);
        assert!((big[0] - 0.5).abs() < 1e-6, "no overflow: {big:?}");

        // Uniform logits → uniform distribution.
        let u = softmax_n(&vec![0.0; V]);
        assert!(u.iter().all(|&x| (x - 1.0 / V as f32).abs() < 1e-6));

        // score_n is a per-class dot product.
        let w = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        assert_eq!(score_n(&w, &[3.0, 5.0]), vec![3.0, 10.0]);

        // argmax_first breaks ties toward the lower index (matching the 2-class `s1 > s0` rule).
        assert_eq!(argmax_first(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(argmax_first(&[1.0, 5.0, 5.0]), 1);

        // total_variation: 0 when identical; 1/2 when a 50/50 fork collapses onto one branch;
        // 1/3 when [1]'s 67/33 collapses. Both figures are quoted in the spec's metrics.
        let fork = vec![0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(total_variation(&fork, &fork).abs() < 1e-6);
        let collapsed = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((total_variation(&fork, &collapsed) - 0.5).abs() < 1e-6);
        let skew = conditional(4, &[0]); // {2: 2/3, 4: 1/3}
        let onto2 = {
            let mut v = vec![0.0; V];
            v[1] = 1.0;
            v
        };
        assert!((total_variation(&skew, &onto2) - 1.0 / 3.0).abs() < 1e-5);

        // For a deterministic (one-hot) truth, 1 - TV is exactly the mass on the target. This is the
        // identity the checkpoint scalar rests on.
        let truth = conditional(4, &[0, 1, 2]); // one-hot on token 3
        let q = softmax_n(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((1.0 - total_variation(&truth, &q) - q[3]).abs() < 1e-5);
    }

    #[test]
    fn seq_conditionals_correct() {
        // Prefix enumeration: 9 / 12 / 15 distinct prefixes.
        assert_eq!(prefixes(4).len(), 9);
        assert_eq!(prefixes(5).len(), 12);
        assert_eq!(prefixes(6).len(), 15);

        // The two forks, in closed form.
        let c1 = conditional(4, &[0]);
        assert!((c1[1] - 2.0 / 3.0).abs() < 1e-6, "[1] → 2 with p=2/3");
        assert!((c1[3] - 1.0 / 3.0).abs() < 1e-6, "[1] → 4 with p=1/3");
        let c12 = conditional(4, &[0, 1]);
        assert!((c12[2] - 0.5).abs() < 1e-6, "[1,2] → 3 with p=1/2");
        assert!((c12[3] - 0.5).abs() < 1e-6, "[1,2] → 4 with p=1/2");

        // Deterministic prefixes, including the disambiguation pair.
        assert_eq!(conditional(4, &[0, 1, 2])[3], 1.0, "[1,2,3] → 4");
        assert_eq!(conditional(4, &[1, 1, 2])[4], 1.0, "[2,2,3] → 5");
        assert_eq!(conditional(4, &[0, 3, 7])[8], 1.0, "[1,4,8] → 16");

        // Every conditional is a distribution.
        for set_size in [4, 5, 6] {
            for p in prefixes(set_size) {
                let s: f32 = conditional(set_size, &p).iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "conditional sums to 1 for {p:?}");
            }
        }

        // The family grows 2 → 3 → 4-way, all sharing the (2,3) suffix.
        assert_eq!(family(4).len(), 2);
        assert_eq!(family(5).len(), 3);
        assert_eq!(family(6).len(), 4);

        // Markov-2 ceiling on the family is exactly 1/k — the control the whole task rests on.
        for (set_size, k) in [(4, 2.0), (5, 3.0), (6, 4.0)] {
            let m2 = markov_k_accuracy(set_size, 2, &family(set_size));
            assert!((m2 - 1.0 / k).abs() < 1e-6, "Markov-2 family ceiling is 1/{k} for set {set_size}, got {m2}");
        }

        // Markov-1 is never better than Markov-2 → Markov-2 is the discriminating control.
        for set_size in [4, 5, 6] {
            let m1 = markov_k_accuracy(set_size, 1, &family(set_size));
            let m2 = markov_k_accuracy(set_size, 2, &family(set_size));
            assert!(m1 <= m2 + 1e-6, "Markov-1 ({m1}) must not beat Markov-2 ({m2}) for set {set_size}");
        }

        // Markov-3 sees the whole prefix, so it is full memory: 100% on the family.
        for set_size in [4, 5, 6] {
            let m3 = markov_k_accuracy(set_size, 3, &family(set_size));
            assert!((m3 - 1.0).abs() < 1e-6, "Markov-3 == full memory for set {set_size}, got {m3}");
        }

        // seq_task only ever emits a real (prefix, target) pair from the set.
        let task = seq_task(4);
        for t in 0..500 {
            let (prefix, target) = task(7, t);
            assert!((1..=MAX_PREFIX).contains(&prefix.len()), "prefix length in 1..=3, got {prefix:?}");
            let cond = conditional(4, &prefix);
            assert!(cond[target] > 0.0, "target {target} must be reachable from {prefix:?}");
        }
    }
}
