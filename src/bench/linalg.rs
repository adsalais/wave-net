//! `linalg` — the small f64 linear algebra the bench readouts need: an LU solve (Gaussian
//! elimination with partial pivoting, factored once and reused across right-hand sides) and the
//! normal-equation builders `XᵀX` and `Xᵀy`. Single-threaded and deterministic.

/// LU factorization with partial pivoting of a square matrix; reusable across right-hand sides.
pub struct Lu {
    lu: Vec<Vec<f64>>, // L (unit diag, below) and U (on/above) packed
    piv: Vec<usize>,   // row permutation
    n: usize,
}

impl Lu {
    /// Factor a square matrix. Panics if it is empty or exactly singular.
    pub fn factor(mut a: Vec<Vec<f64>>) -> Lu {
        let n = a.len();
        assert!(n > 0 && a.iter().all(|r| r.len() == n), "matrix must be square and non-empty");
        let mut piv: Vec<usize> = (0..n).collect();
        for col in 0..n {
            let mut pmax = col;
            let mut vmax = a[col][col].abs();
            for r in (col + 1)..n {
                if a[r][col].abs() > vmax {
                    vmax = a[r][col].abs();
                    pmax = r;
                }
            }
            assert!(vmax > 0.0, "singular matrix");
            if pmax != col {
                a.swap(col, pmax);
                piv.swap(col, pmax);
            }
            let pivot = a[col][col];
            for r in (col + 1)..n {
                let f = a[r][col] / pivot;
                a[r][col] = f; // store multiplier
                for c in (col + 1)..n {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
        Lu { lu: a, piv, n }
    }

    /// Solve `A x = b`.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = b[self.piv[i]];
        }
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.lu[i][j] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.lu[i][j] * x[j];
            }
            x[i] /= self.lu[i][i];
        }
        x
    }
}

/// `Xᵀ X` — square, dimension = number of columns of `x`.
pub fn xt_x(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let mut a = vec![vec![0.0; d]; d];
    for row in x {
        for i in 0..d {
            let ri = row[i];
            for j in 0..d {
                a[i][j] += ri * row[j];
            }
        }
    }
    a
}

/// `Xᵀ y`.
pub fn xt_y(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let mut v = vec![0.0; d];
    for (row, &yi) in x.iter().zip(y) {
        for i in 0..d {
            v[i] += row[i] * yi;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solves_known_system() {
        // [[2,1],[1,3]] x = [3,5] -> x = [0.8, 1.4]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let lu = Lu::factor(a);
        let x = lu.solve(&[3.0, 5.0]);
        assert!((x[0] - 0.8).abs() < 1e-9 && (x[1] - 1.4).abs() < 1e-9, "got {x:?}");
    }
}
