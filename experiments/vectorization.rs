// Standalone code-generation experiment, not a replacement BM25 engine.
#[no_mangle]
pub fn dense_accumulate(scores: &mut [f32], weights: &[f32], factor: f32) {
    assert_eq!(scores.len(), weights.len());
    for (score, &weight) in scores.iter_mut().zip(weights) {
        *score += weight * factor;
    }
}

#[no_mangle]
pub fn scatter_accumulate(scores: &mut [f32], ids: &[u32], weights: &[f32], factor: f32) {
    assert_eq!(ids.len(), weights.len());
    for (&id, &weight) in ids.iter().zip(weights) {
        scores[id as usize] += weight * factor;
    }
}

#[no_mangle]
pub fn normalize_tf(out: &mut [f32], tf: &[f32], norm: &[f32], k1: f32) {
    assert_eq!(out.len(), tf.len());
    assert_eq!(tf.len(), norm.len());
    for ((out, &tf), &norm) in out.iter_mut().zip(tf).zip(norm) {
        *out = (tf * (k1 + 1.0)) / (tf + norm);
    }
}

#[no_mangle]
pub fn strict_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        sum += a * b;
    }
    sum
}

#[no_mangle]
pub fn four_lane_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sums = [0.0; 4];
    let mut ac = a.chunks_exact(4);
    let mut bc = b.chunks_exact(4);
    for (a, b) in ac.by_ref().zip(bc.by_ref()) {
        for j in 0..4 {
            sums[j] += a[j] * b[j];
        }
    }
    let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for (&a, &b) in ac.remainder().iter().zip(bc.remainder()) {
        sum += a * b;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kernels_match_references_including_tails_and_repeated_ids() {
        for n in [0, 1, 3, 4, 5, 31, 32, 33, 129] {
            let weights: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.25).collect();
            let mut scores = vec![0.5; n];
            dense_accumulate(&mut scores, &weights, 1.5);
            for (s, w) in scores.iter().zip(&weights) { assert_eq!(*s, 0.5 + w * 1.5); }
            let mut sparse_scores = vec![0.5; 11];
            let ids: Vec<u32> = (0..n).map(|i| (i % 11) as u32).collect();
            scatter_accumulate(&mut sparse_scores, &ids, &weights, 1.5);
            let mut expected = vec![0.5; 11];
            for (id, w) in ids.iter().zip(&weights) { expected[*id as usize] += w * 1.5; }
            assert_eq!(expected, sparse_scores);
            let norm = vec![0.75; n];
            let mut normalized = vec![0.0; n];
            normalize_tf(&mut normalized, &weights, &norm, 1.2);
            for (out, tf) in normalized.iter().zip(&weights) { assert_eq!(*out, (tf * (1.2 + 1.0)) / (tf + 0.75)); }
            let expected_dot: f32 = weights.iter().zip(&norm).fold(0.0, |s, (a,b)| s + a*b);
            assert_eq!(strict_dot(&weights, &norm), expected_dot);
            assert!((four_lane_dot(&weights, &norm) - expected_dot).abs() <= 1e-5 * (1.0 + expected_dot.abs()));
        }
    }
}

#[no_mangle]
pub fn algebraic_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0_f32, |sum, (&a, &b)| sum.algebraic_add(a * b))
}

#[no_mangle]
pub fn dense_fma(scores: &mut [f32], weights: &[f32], factor: f32) {
    assert_eq!(scores.len(), weights.len());
    for (score, &weight) in scores.iter_mut().zip(weights) {
        *score = weight.mul_add(factor, *score);
    }
}

#[cfg(test)]
mod relaxed_tests {
    use super::*;
    #[test]
    fn relaxed_kernels_match_f64_reference_on_positive_bm25_like_inputs() {
        for n in [0,1,3,4,5,31,32,33,129,4097] {
            let a: Vec<f32> = (0..n).map(|i| ((i * 37 % 997) as f32 + 1.0) / 997.0).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i * 13 % 503) as f32 + 1.0) / 251.0).collect();
            let reference: f64 = a.iter().zip(&b).map(|(&a,&b)| a as f64 * b as f64).sum();
            assert!((algebraic_dot(&a,&b) as f64-reference).abs() <= 1e-5*(1.0+reference.abs()));
            let mut scores=b.clone();
            dense_fma(&mut scores,&a,1.2);
            for ((&score,&weight), &initial) in scores.iter().zip(&a).zip(&b) {
                assert_eq!(score, weight.mul_add(1.2,initial));
            }
        }
    }
}
