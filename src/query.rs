use crate::{embedder::DefaultEmbeddingSpace, Embedding, TokenEmbedding};
use std::{collections::HashMap, hash::Hash};

const HASH_THRESHOLD: usize = 128;

/// Unique query terms and their nonnegative boosts (term frequency by default).
/// Query weights are independent of document-length normalization.
/// Terms retain first-occurrence order even when hash-assisted lookup is used.
///
/// ```
/// use bm_25::{Query, TokenEmbedding};
/// let mut query = Query::<u32>::default();
/// query.push(7, 1.0);
/// query.extend([TokenEmbedding::new(7, 2.0), TokenEmbedding::new(9, 1.0)]);
/// assert_eq!(query.terms()[0].value, 3.0);
/// query.clear(); // Retains term and lookup capacity for the next query.
/// query.push(42, 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct Query<D = DefaultEmbeddingSpace> {
    pub(crate) terms: Vec<TokenEmbedding<D>>,
    positions: Option<HashMap<D, usize>>,
}

impl<D: PartialEq> PartialEq for Query<D> {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl<D> Default for Query<D> {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            positions: None,
        }
    }
}

impl<D: Eq + Hash + Clone> Query<D> {
    /// Builds a query, combining duplicate term boosts in first-occurrence order.
    /// Panics for negative, nonfinite, or overflowing combined boosts.
    pub fn new(terms: impl IntoIterator<Item = TokenEmbedding<D>>) -> Self {
        let mut query = Self::default();
        query.extend(terms);
        query
    }

    /// Adds a term or increases its boost, preserving first-occurrence order.
    /// Small queries use a linear lookup; large queries use a retained hash table.
    /// Panics for negative/nonfinite weights or a nonfinite combined boost.
    pub fn push(&mut self, index: D, value: f32) {
        assert!(
            value.is_finite() && value >= 0.0,
            "query boosts must be finite and nonnegative"
        );
        if value == 0.0 {
            return;
        }
        let position = if self.terms.len() < HASH_THRESHOLD {
            self.terms.iter().position(|t| t.index == index)
        } else {
            let positions = self.positions.get_or_insert_with(HashMap::new);
            if positions.is_empty() {
                positions.extend(
                    self.terms
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (t.index.clone(), i)),
                );
            }
            positions.get(&index).copied()
        };
        if let Some(position) = position {
            let combined = self.terms[position].value + value;
            assert!(combined.is_finite(), "query boost overflow");
            self.terms[position].value = combined;
        } else {
            if let Some(positions) = self.positions.as_mut().filter(|p| !p.is_empty()) {
                positions.insert(index.clone(), self.terms.len());
            }
            self.terms.push(TokenEmbedding { index, value });
        }
    }

    /// Adds weighted terms using the same validation and order as [`Self::push`].
    pub fn extend(&mut self, terms: impl IntoIterator<Item = TokenEmbedding<D>>) {
        for term in terms {
            self.push(term.index, term.value);
        }
    }
}

impl<D> Query<D> {
    /// Returns the unique query terms in first-occurrence order.
    pub fn terms(&self) -> &[TokenEmbedding<D>] {
        &self.terms
    }
    /// Clears the query while retaining term and lookup allocations.
    pub fn clear(&mut self) {
        self.terms.clear();
        if let Some(positions) = &mut self.positions {
            positions.clear();
        }
    }
}

impl<D: Eq + Hash + Clone> From<&Embedding<D>> for Query<D> {
    /// Interprets each embedding index as one query occurrence, ignoring document weights.
    fn from(embedding: &Embedding<D>) -> Self {
        Self::new(embedding.iter().map(|t| TokenEmbedding {
            index: t.index.clone(),
            value: 1.0,
        }))
    }
}
