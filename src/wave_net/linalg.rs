//! Small dense linear algebra for ridge readouts: Cholesky solve of the normal equations.
//! Lifted from the `params_study` example so the library and its examples share one copy.

/// Solve `A X = B` for symmetric positive-definite `A` (in place) via Cholesky. `A` is the
/// coefficient matrix, `B` its right-hand side columns; returns `X` with the same column count.
pub fn cholesky_solve(mut a: Vec<Vec<f64>>, b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            if i == j {
                a[i][i] = s.max(1e-12).sqrt();
            } else {
                a[i][j] = s / a[j][j];
            }
        }
    }
    let m = b[0].len();
    let mut x = vec![vec![0.0f64; m]; n];
    for col in 0..m {
        let mut z = vec![0.0f64; n];
        for i in 0..n {
            let mut s = b[i][col];
            for k in 0..i {
                s -= a[i][k] * z[k];
            }
            z[i] = s / a[i][i];
        }
        for i in (0..n).rev() {
            let mut s = z[i];
            for k in (i + 1)..n {
                s -= a[k][i] * x[k][col];
            }
            x[i][col] = s / a[i][i];
        }
    }
    x
}

/// Ridge regression: fit `w` minimizing `||Xw - y||² + λ||w||²` via the normal equations
/// `(XᵀX + λI) w = Xᵀy`. Rows of `x` are samples (including any bias column), `y` the targets.
pub fn ridge_fit(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let d = x[0].len();
    let mut a = vec![vec![0.0f64; d]; d];
    let mut b = vec![vec![0.0f64; 1]; d];
    for (xi, &yi) in x.iter().zip(y) {
        for i in 0..d {
            b[i][0] += xi[i] * yi;
            for j in 0..d {
                a[i][j] += xi[i] * xi[j];
            }
        }
    }
    for i in 0..d {
        a[i][i] += lambda;
    }
    cholesky_solve(a, &b).into_iter().map(|r| r[0]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_recovers_a_known_linear_map() {
        // y = 3x + 0.5; features [x, bias]. With negligible λ, ridge recovers (3, 0.5).
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64, 1.0]).collect();
        let y: Vec<f64> = (0..8).map(|i| 3.0 * i as f64 + 0.5).collect();
        let w = ridge_fit(&x, &y, 1e-9);
        assert!((w[0] - 3.0).abs() < 1e-3, "slope {} != 3", w[0]);
        assert!((w[1] - 0.5).abs() < 1e-3, "intercept {} != 0.5", w[1]);
    }

    #[test]
    fn cholesky_solves_spd_system() {
        // A = [[4,2],[2,3]], b = [1, 1]ᵀ → x = [1/8, 1/4].
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let b = vec![vec![1.0], vec![1.0]];
        let x = cholesky_solve(a, &b);
        assert!((x[0][0] - 0.125).abs() < 1e-9, "x0 = {}", x[0][0]);
        assert!((x[1][0] - 0.25).abs() < 1e-9, "x1 = {}", x[1][0]);
    }
}
