/// Reusable storage for normalized text. One buffer may be reused across documents.
#[derive(Default, Debug)]
pub struct TokenizerScratch {
    #[cfg(feature = "default_tokenizer")]
    pub(crate) normalized: String,
}

/// Splits text into tokens. Override `for_each_token` to avoid owned per-token strings.
pub trait Tokenizer {
    /// Convenience interface returning owned tokens.
    fn tokenize<'a>(&'a self, input_text: &'a str) -> impl Iterator<Item = String> + 'a;

    /// Visits temporary token slices. The callback must not retain these slices.
    /// The default implementation adapts existing owned tokenizers.
    fn for_each_token(
        &self,
        input: &str,
        _scratch: &mut TokenizerScratch,
        mut visit: impl FnMut(&str),
    ) {
        for token in self.tokenize(input) {
            visit(&token);
        }
    }
}
