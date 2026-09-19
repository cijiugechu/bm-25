//! Block scheduling is separate from scoring. Each batch contains only matching
//! term blocks, in query order, so bounds and scores share the same addition order.
use crate::index::{Block, PreparedTerm};
use std::{cmp::Reverse, collections::BinaryHeap};

/// How sorted posting-block streams are merged. Does not change scoring semantics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TraversalMode {
    /// Choose scanning for short/high-overlap queries and a heap for sparse fan-out.
    #[default]
    Auto,
    /// Scan all query cursors for each document block.
    Scan,
    /// Maintain a min-heap of active cursors; useful for many disjoint streams.
    Heap,
}

#[derive(Debug)]
struct Cursor {
    next: usize,
    end: usize,
    factor: f32,
}

#[derive(Debug)]
pub(crate) struct Match {
    pub(crate) block: usize,
    pub(crate) factor: f32,
}

#[derive(Debug, Default)]
pub(crate) struct Traversal {
    cursors: Vec<Cursor>,
    // Secondary query ordinal preserves floating-point accumulation order.
    heap: BinaryHeap<Reverse<(u32, usize)>>,
    pub(crate) matches: Vec<Match>,
    mode: TraversalMode,
    next_block: Option<u32>,
}

impl Traversal {
    pub(crate) fn reset(
        &mut self,
        terms: &[PreparedTerm],
        blocks: &[Block],
        mode: TraversalMode,
    ) -> TraversalMode {
        self.cursors.clear();
        self.heap.clear();
        self.matches.clear();
        self.cursors.extend(terms.iter().map(|t| Cursor {
            next: t.blocks.start,
            end: t.blocks.end,
            factor: t.factor,
        }));
        self.next_block = self.cursors.iter().map(|c| blocks[c.next].doc_block).min();
        self.mode = match mode {
            TraversalMode::Auto if terms.len() > 16 => {
                let first = terms
                    .iter()
                    .map(|t| blocks[t.blocks.start].doc_block)
                    .min()
                    .unwrap_or(0);
                let last = terms
                    .iter()
                    .map(|t| blocks[t.blocks.end - 1].doc_block)
                    .max()
                    .unwrap_or(0);
                let count: usize = terms.iter().map(|t| t.blocks.len()).sum();
                // Estimate overlap without walking postings. A heuristic, not a
                // guarantee: callers can benchmark and override either strategy.
                let span = (last - first) as f64 + 1.0;
                if count as f64 / span < terms.len() as f64 / 4.0 {
                    TraversalMode::Heap
                } else {
                    TraversalMode::Scan
                }
            }
            TraversalMode::Auto => TraversalMode::Scan,
            explicit => explicit,
        };
        if self.mode == TraversalMode::Heap {
            self.heap.extend(
                self.cursors
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Reverse((blocks[c.next].doc_block, i))),
            );
        }
        self.mode
    }

    pub(crate) fn next(&mut self, blocks: &[Block]) -> Option<u32> {
        self.matches.clear();
        match self.mode {
            TraversalMode::Heap => {
                let &Reverse((doc_block, _)) = self.heap.peek()?;
                while let Some(mut head) = self.heap.peek_mut() {
                    let Reverse((block, ordinal)) = *head;
                    if block != doc_block {
                        break;
                    }
                    let cursor = &mut self.cursors[ordinal];
                    self.matches.push(Match {
                        block: cursor.next,
                        factor: cursor.factor,
                    });
                    cursor.next += 1;
                    if cursor.next == cursor.end {
                        std::collections::binary_heap::PeekMut::pop(head);
                    } else {
                        *head = Reverse((blocks[cursor.next].doc_block, ordinal));
                    }
                }
                Some(doc_block)
            }
            _ => {
                let doc_block = self.next_block.take()?;
                // Document IDs are u32, so a 128-document block can never have
                // ordinal u32::MAX. Cache the next minimum while advancing.
                let mut next_block = u32::MAX;
                for cursor in &mut self.cursors {
                    if cursor.next == cursor.end {
                        continue;
                    }
                    let head = blocks[cursor.next].doc_block;
                    if head == doc_block {
                        self.matches.push(Match {
                            block: cursor.next,
                            factor: cursor.factor,
                        });
                        cursor.next += 1;
                        if cursor.next < cursor.end {
                            next_block = next_block.min(blocks[cursor.next].doc_block);
                        }
                    } else {
                        next_block = next_block.min(head);
                    }
                }
                self.next_block = (next_block != u32::MAX).then_some(next_block);
                Some(doc_block)
            }
        }
    }
}
