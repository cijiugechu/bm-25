use crate::index::Hit;
use std::{cmp::Ordering, collections::BinaryHeap};

// Largest heap element is the worst retained hit. Equal scores prefer the lower DocId.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate(pub(crate) Hit);

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then(self.0.doc_id.cmp(&other.0.doc_id))
    }
}

pub(crate) fn retain(heap: &mut BinaryHeap<Candidate>, hit: Hit, limit: usize) {
    let candidate = Candidate(hit);
    if heap.len() < limit {
        heap.push(candidate);
    } else if let Some(mut worst) = heap.peek_mut() {
        if candidate < *worst {
            *worst = candidate;
        }
    }
}

pub(crate) fn sort(hits: &mut [Hit]) {
    hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
}
