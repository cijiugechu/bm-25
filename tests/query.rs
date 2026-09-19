use bm_25::{Query, TokenEmbedding as Token};

#[test]
fn query_reuse_crosses_hash_threshold_without_changing_order_or_boosts() {
    let mut query = Query::default();
    for count in [1024, 2, 128, 129, 0, 127] {
        query.clear();
        query.extend(
            (0..count)
                .rev()
                .map(|i| Token::new(format!("term-{i}"), 1.0)),
        );
        for i in 0..count {
            query.push(format!("term-{i}"), 0.5);
        }
        query.push("omitted".into(), 0.0);
        let expected: Vec<_> = (0..count)
            .rev()
            .map(|i| Token::new(format!("term-{i}"), 1.5))
            .collect();
        assert_eq!(query.terms(), expected);
        assert_eq!(query, Query::new(expected));
        let mut cloned = query.clone();
        cloned.push("new".into(), 3.0);
        assert_eq!(cloned.terms().len(), query.terms().len() + 1);
    }
}

#[test]
fn hash_collisions_do_not_merge_distinct_terms() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Key(u32);
    impl std::hash::Hash for Key {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            state.write_u8(0);
        }
    }
    let mut query = Query::new((0..160).map(|i| Token::new(Key(i), 1.0)));
    for i in (0..160).rev() {
        query.push(Key(i), 2.0);
    }
    assert_eq!(
        query.terms(),
        (0..160)
            .map(|i| Token::new(Key(i), 3.0))
            .collect::<Vec<_>>()
    );
}

#[test]
fn invalid_push_preserves_valid_query_and_duplicate_addition_order() {
    let mut query = Query::new((0..160).map(|i| Token::new(i, 1.0)));
    query.push(0, f32::MAX);
    let before = query.clone();
    for (key, weight) in [
        (0, f32::MAX),
        (50, -1.0),
        (50, f32::NAN),
        (50, f32::INFINITY),
    ] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| query.push(key, weight)))
                .is_err()
        );
        assert_eq!(query, before);
    }
    query.clear();
    query.push(0, 16_777_216.0);
    query.push(0, 1.0);
    query.push(0, 1.0);
    assert_eq!(query.terms()[0].value, 16_777_216.0);
}
