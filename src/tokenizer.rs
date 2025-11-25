/// A tokenizer splits text into a sequence of tokens. Implement this trait to use this crate with
/// your own tokenizer.
pub trait Tokenizer {
    /// Tokenizes the input text.
    fn tokenize<'a>(&'a self, input_text: &'a str) -> impl Iterator<Item = String> + 'a;
}
