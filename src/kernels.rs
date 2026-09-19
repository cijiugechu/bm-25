//! Contiguous, independent output lanes let LLVM vectorize without unsafe code.

pub(crate) fn accumulate(scores: &mut [f32], weights: &[f32], factor: f32) {
    assert_eq!(scores.len(), weights.len());
    for (score, &weight) in scores.iter_mut().zip(weights) {
        *score += weight * factor;
    }
}

pub(crate) fn accumulate_relaxed(scores: &mut [f32], weights: &[f32], factor: f32) {
    assert_eq!(scores.len(), weights.len());
    for (score, &weight) in scores.iter_mut().zip(weights) {
        *score = score.algebraic_add(weight * factor);
    }
}
