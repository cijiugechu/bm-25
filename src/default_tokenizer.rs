use bitflags::bitflags;
use cached::proc_macro::cached;
use rust_stemmers::{Algorithm as StemmingAlgorithm, Stemmer};
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::{self, Debug},
};
use stop_words::LANGUAGE as StopWordLanguage;
#[cfg(feature = "language_detection")]
use whichlang::Lang as DetectedLanguage;

use crate::tokenizer::Tokenizer;

/// Languages supported by the tokenizer.
#[allow(missing_docs)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Language {
    Arabic,
    Danish,
    Dutch,
    English,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

/// The language mode used by the tokenizer. This determines the algorithm used for stemming and
/// the dictionary of stopwords. This enum is non-exhaustive as the `Detect` variant is only
/// available when the `language_detection` feature is enabled.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LanguageMode {
    /// Automatically detect the language. Note that this adds a small performance overhead.
    #[cfg(feature = "language_detection")]
    Detect,
    /// Use a fixed language.
    Fixed(Language),
}

impl Default for LanguageMode {
    fn default() -> Self {
        LanguageMode::Fixed(Language::English)
    }
}

impl From<Language> for LanguageMode {
    fn from(language: Language) -> Self {
        LanguageMode::Fixed(language)
    }
}

#[cfg(feature = "language_detection")]
impl TryFrom<DetectedLanguage> for Language {
    type Error = ();

    fn try_from(detected_language: DetectedLanguage) -> Result<Self, Self::Error> {
        match detected_language {
            DetectedLanguage::Ara => Ok(Language::Arabic),
            DetectedLanguage::Cmn => Err(()),
            DetectedLanguage::Deu => Ok(Language::German),
            DetectedLanguage::Eng => Ok(Language::English),
            DetectedLanguage::Fra => Ok(Language::French),
            DetectedLanguage::Hin => Err(()),
            DetectedLanguage::Ita => Ok(Language::Italian),
            DetectedLanguage::Jpn => Err(()),
            DetectedLanguage::Kor => Err(()),
            DetectedLanguage::Nld => Ok(Language::Dutch),
            DetectedLanguage::Por => Ok(Language::Portuguese),
            DetectedLanguage::Rus => Ok(Language::Russian),
            DetectedLanguage::Spa => Ok(Language::Spanish),
            DetectedLanguage::Swe => Ok(Language::Swedish),
            DetectedLanguage::Tur => Ok(Language::Turkish),
            DetectedLanguage::Vie => Err(()),
        }
    }
}

impl From<&Language> for StemmingAlgorithm {
    fn from(language: &Language) -> Self {
        match language {
            Language::Arabic => StemmingAlgorithm::Arabic,
            Language::Danish => StemmingAlgorithm::Danish,
            Language::Dutch => StemmingAlgorithm::Dutch,
            Language::English => StemmingAlgorithm::English,
            Language::French => StemmingAlgorithm::French,
            Language::German => StemmingAlgorithm::German,
            Language::Greek => StemmingAlgorithm::Greek,
            Language::Hungarian => StemmingAlgorithm::Hungarian,
            Language::Italian => StemmingAlgorithm::Italian,
            Language::Norwegian => StemmingAlgorithm::Norwegian,
            Language::Portuguese => StemmingAlgorithm::Portuguese,
            Language::Romanian => StemmingAlgorithm::Romanian,
            Language::Russian => StemmingAlgorithm::Russian,
            Language::Spanish => StemmingAlgorithm::Spanish,
            Language::Swedish => StemmingAlgorithm::Swedish,
            Language::Tamil => StemmingAlgorithm::Tamil,
            Language::Turkish => StemmingAlgorithm::Turkish,
        }
    }
}

impl TryFrom<&Language> for StopWordLanguage {
    type Error = ();

    fn try_from(language: &Language) -> Result<Self, Self::Error> {
        match language {
            Language::Arabic => Ok(StopWordLanguage::Arabic),
            Language::Danish => Ok(StopWordLanguage::Danish),
            Language::Dutch => Ok(StopWordLanguage::Dutch),
            Language::English => Ok(StopWordLanguage::English),
            Language::French => Ok(StopWordLanguage::French),
            Language::German => Ok(StopWordLanguage::German),
            Language::Greek => Ok(StopWordLanguage::Greek),
            Language::Hungarian => Ok(StopWordLanguage::Hungarian),
            Language::Italian => Ok(StopWordLanguage::Italian),
            Language::Norwegian => Ok(StopWordLanguage::Norwegian),
            Language::Portuguese => Ok(StopWordLanguage::Portuguese),
            Language::Romanian => Ok(StopWordLanguage::Romanian),
            Language::Russian => Ok(StopWordLanguage::Russian),
            Language::Spanish => Ok(StopWordLanguage::Spanish),
            Language::Swedish => Ok(StopWordLanguage::Swedish),
            Language::Tamil => Err(()),
            Language::Turkish => Ok(StopWordLanguage::Turkish),
        }
    }
}

fn normalize(text: &str) -> Cow<'_, str> {
    deunicode::deunicode_with_tofu_cow(text, "[?]")
}

#[cached(size = 16)]
fn get_stopwords(language: Language, normalized: bool) -> HashSet<String> {
    match TryInto::<StopWordLanguage>::try_into(&language) {
        Err(_) => HashSet::new(),
        Ok(lang) => stop_words::get(lang)
            .iter()
            .map(|w| match normalized {
                true => normalize(w).into(),
                false => w.to_string(),
            })
            .collect(),
    }
}

fn get_stemmer(language: &Language) -> Stemmer {
    Stemmer::create(language.into())
}

struct WordIter {
    text: String,
    offset: usize,
}

impl WordIter {
    fn new(text: String) -> Self {
        WordIter { text, offset: 0 }
    }
}

impl Iterator for WordIter {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        use unicode_segmentation::UnicodeSegmentation;

        let slice = &self.text[self.offset..];
        let mut words = slice.unicode_word_indices();
        let (relative_idx, word) = words.next()?;
        self.offset += relative_idx + word.len();
        Some(word.to_string())
    }
}

struct TokenIterBorrowed<'a> {
    word_iter: WordIter,
    stopwords: &'a HashSet<String>,
    stemmer: Option<&'a Stemmer>,
}

impl<'a> Iterator for TokenIterBorrowed<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let token = self.word_iter.next()?;
            if self.stopwords.contains(&token) {
                continue;
            }
            return Some(match self.stemmer {
                Some(stemmer) => stemmer.stem(&token).to_string(),
                None => token,
            });
        }
    }
}

#[cfg(feature = "language_detection")]
struct TokenIterOwned {
    word_iter: WordIter,
    stopwords: HashSet<String>,
    stemmer: Option<Stemmer>,
}

#[cfg(feature = "language_detection")]
impl Iterator for TokenIterOwned {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let token = self.word_iter.next()?;
            if self.stopwords.contains(&token) {
                continue;
            }
            return Some(match &self.stemmer {
                Some(stemmer) => stemmer.stem(&token).to_string(),
                None => token,
            });
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct Settings: u8 {
        const NORMALIZATION = 1 << 0;
        const STEMMING = 1 << 1;
        const STOPWORDS = 1 << 2;
    }
}

impl Settings {
    fn new(stemming: bool, stopwords: bool, normalization: bool) -> Self {
        Settings::from_bits_retain(
            normalization as u8 * Settings::NORMALIZATION.bits()
                | stemming as u8 * Settings::STEMMING.bits()
                | stopwords as u8 * Settings::STOPWORDS.bits(),
        )
    }

    fn normalization_enabled(self) -> bool {
        self.contains(Settings::NORMALIZATION)
    }

    fn stemming_enabled(self) -> bool {
        self.contains(Settings::STEMMING)
    }

    fn stopwords_enabled(self) -> bool {
        self.contains(Settings::STOPWORDS)
    }
}

struct Components {
    settings: Settings,
    normalizer: fn(&str) -> Cow<str>,
    stemmer: Option<Stemmer>,
    stopwords: HashSet<String>,
}

impl Components {
    fn new(settings: Settings, language: Option<&Language>) -> Self {
        let stemmer = language.and_then(|lang| {
            if settings.stemming_enabled() {
                Some(get_stemmer(lang))
            } else {
                None
            }
        });
        let stopwords = language.map_or_else(HashSet::new, |lang| {
            if settings.stopwords_enabled() {
                get_stopwords(lang.clone(), settings.normalization_enabled())
            } else {
                HashSet::new()
            }
        });
        let normalizer: fn(&str) -> Cow<str> = match settings.normalization_enabled() {
            true => normalize,
            false => |text: &str| Cow::from(text),
        };
        Self {
            settings,
            stemmer,
            stopwords,
            normalizer,
        }
    }
}

#[non_exhaustive]
enum Resources {
    Static(Components),
    #[cfg(feature = "language_detection")]
    Dynamic(Settings),
}

pub struct DefaultTokenizer {
    resources: Resources,
}

impl Debug for DefaultTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let settings = match &self.resources {
            Resources::Static(components) => components.settings,
            #[cfg(feature = "language_detection")]
            Resources::Dynamic(settings) => *settings,
        };
        write!(f, "DefaultTokenizer({settings:?})")
    }
}

impl DefaultTokenizer {
    /// Creates a new `DefaultTokenizer` with the given `LanguageMode`.
    pub fn new(language_mode: impl Into<LanguageMode>) -> DefaultTokenizer {
        Self::builder().language_mode(language_mode).build()
    }

    /// Creates a new `DefaultTokenizerBuilder` that you can use to customise the tokenizer.
    pub fn builder() -> DefaultTokenizerBuilder {
        DefaultTokenizerBuilder::new()
    }

    fn _new(
        language_mode: impl Into<LanguageMode>,
        normalization: bool,
        stemming: bool,
        stopwords: bool,
    ) -> DefaultTokenizer {
        let language_mode = &language_mode.into();
        let settings = Settings::new(stemming, stopwords, normalization);
        let resources = match language_mode {
            #[cfg(feature = "language_detection")]
            LanguageMode::Detect => Resources::Dynamic(settings),
            LanguageMode::Fixed(lang) => Resources::Static(Components::new(settings, Some(lang))),
        };
        DefaultTokenizer { resources }
    }

    #[cfg(feature = "language_detection")]
    fn detect_language(text: &str) -> Option<Language> {
        Language::try_from(whichlang::detect_language(text)).ok()
    }

    fn tokenize<'a>(&'a self, input_text: &'a str) -> impl Iterator<Item = String> + 'a {
        enum TokenStream<'a> {
            Borrowed(TokenIterBorrowed<'a>),
            #[cfg(feature = "language_detection")]
            Owned(TokenIterOwned),
        }

        impl<'a> Iterator for TokenStream<'a> {
            type Item = String;

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    TokenStream::Borrowed(iter) => iter.next(),
                    #[cfg(feature = "language_detection")]
                    TokenStream::Owned(iter) => iter.next(),
                }
            }
        }

        let make_word_iter = |input: &str, normalizer: fn(&str) -> Cow<str>| {
            WordIter::new(normalizer(input).to_lowercase())
        };

        match &self.resources {
            Resources::Static(components) => TokenStream::Borrowed(TokenIterBorrowed {
                word_iter: make_word_iter(input_text, components.normalizer),
                stopwords: &components.stopwords,
                stemmer: components.stemmer.as_ref(),
            }),
            #[cfg(feature = "language_detection")]
            Resources::Dynamic(settings) => {
                let detected_language = Self::detect_language(input_text);
                let components = Components::new(*settings, detected_language.as_ref());

                TokenStream::Owned(TokenIterOwned {
                    word_iter: make_word_iter(input_text, components.normalizer),
                    stopwords: components.stopwords,
                    stemmer: components.stemmer,
                })
            }
        }
    }
}

impl Tokenizer for DefaultTokenizer {
    fn tokenize<'a>(&'a self, input_text: &'a str) -> impl Iterator<Item = String> + 'a {
        DefaultTokenizer::tokenize(self, input_text)
    }
}

impl Default for DefaultTokenizer {
    fn default() -> Self {
        DefaultTokenizer::new(LanguageMode::default())
    }
}

pub struct DefaultTokenizerBuilder {
    language_mode: LanguageMode,
    normalization: bool,
    stemming: bool,
    stopwords: bool,
}

impl Default for DefaultTokenizerBuilder {
    fn default() -> Self {
        DefaultTokenizerBuilder::new()
    }
}

impl DefaultTokenizerBuilder {
    /// Creates a new `DefaultTokenizerBuilder`.
    pub fn new() -> DefaultTokenizerBuilder {
        DefaultTokenizerBuilder {
            language_mode: LanguageMode::default(),
            normalization: true,
            stemming: true,
            stopwords: true,
        }
    }

    /// Sets the language mode used by the tokenizer. Default is `Language::English`.
    pub fn language_mode(mut self, language_mode: impl Into<LanguageMode>) -> Self {
        self.language_mode = language_mode.into();
        self
    }

    /// Enables or disables normalization. Normalization converts unicode characters to ASCII.
    /// (With normalization, 'é' -> 'e', '🍕' -> 'pizza', etc.)
    /// Default is `true`.
    pub fn normalization(mut self, normalization: bool) -> Self {
        self.normalization = normalization;
        self
    }

    /// Enables or disables stemming. Stemming reduces words to their root form.
    /// (With stemming, 'running' -> 'run', 'connection' -> 'connect', etc.)
    /// Default is `true`.
    pub fn stemming(mut self, stemming: bool) -> Self {
        self.stemming = stemming;
        self
    }

    /// Enables or disables stopwords. Stopwords are common words that carry little meaning.
    /// (With stopwords, 'the', 'and', 'is', etc. are removed.)
    /// Default is `true`.
    pub fn stopwords(mut self, stopwords: bool) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Builds the `DefaultTokenizer`.
    pub fn build(self) -> DefaultTokenizer {
        DefaultTokenizer::_new(
            self.language_mode,
            self.normalization,
            self.stemming,
            self.stopwords,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::test_data_loader::tests::{read_recipes, Recipe};

    use super::*;

    use insta::assert_debug_snapshot;

    fn tokenize_recipes(recipe_file: &str, language_mode: LanguageMode) -> Vec<Vec<String>> {
        let recipes = read_recipes(recipe_file);

        recipes
            .iter()
            .map(|Recipe { recipe, .. }| {
                let tokenizer = DefaultTokenizer::new(language_mode.clone());
                tokenizer.tokenize(recipe).collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn it_can_tokenize_english() {
        let text = "space station";
        let tokenizer = DefaultTokenizer::new(Language::English);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["space", "station"]);
    }

    #[test]
    fn it_converts_to_lowercase() {
        let text = "SPACE STATION";
        let tokenizer = DefaultTokenizer::new(Language::English);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["space", "station"]);
    }

    #[test]
    fn it_removes_whitespace() {
        let text = "\tspace\r\nstation\n space       station";
        let tokenizer = DefaultTokenizer::new(Language::English);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["space", "station", "space", "station"]);
    }

    #[test]
    fn it_removes_stopwords() {
        let text = "i me my myself we our ours ourselves you you're you've you'll you'd";
        let tokenizer = DefaultTokenizer::new(Language::English);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert!(tokens.is_empty());
    }

    #[test]
    fn it_keeps_numbers() {
        let text = "42 1337 3.14";
        let tokenizer = DefaultTokenizer::new(Language::English);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["42", "1337", "3.14"]);
    }

    #[test]
    fn it_keeps_contracted_words() {
        let text = "can't you're won't let's couldn't've";
        let tokenizer = DefaultTokenizer::builder()
            .language_mode(Language::English)
            .stemming(false)
            .stopwords(false)
            .build();

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(
            tokens,
            vec!["can't", "you're", "won't", "let's", "couldn't've"]
        );
    }

    #[test]
    fn it_removes_punctuation() {
        let test_cases = vec![
            ("space, station!", vec!["space", "station"]),
            ("space,station", vec!["space", "station"]),
            ("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", vec![]),
        ];
        let tokenizer = DefaultTokenizer::new(Language::English);

        for (text, expected) in test_cases {
            let tokens: Vec<_> = tokenizer.tokenize(text).collect();
            assert_eq!(tokens, expected);
        }
    }

    #[test]
    fn it_stems_words() {
        let text = "connection connections connective connected connecting connect";
        let tokenizer = DefaultTokenizer::new(Language::English);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(
            tokens,
            vec!["connect", "connect", "connect", "connect", "connect", "connect"]
        );
    }

    #[test]
    fn it_tokenizes_emojis_as_text() {
        let text = "🍕 🚀 🍋";
        let tokenizer = DefaultTokenizer::new(Language::English);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["pizza", "rocket", "lemon"]);
    }

    #[test]
    fn it_converts_unicode_to_ascii() {
        let text = "gemüse, Gießen";
        let tokenizer = DefaultTokenizer::builder()
            .language_mode(Language::German)
            .stemming(false)
            .build();

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["gemuse", "giessen"]);
    }

    #[test]
    #[cfg(feature = "language_detection")]
    fn it_handles_empty_input() {
        let text = "";
        let tokenizer = DefaultTokenizer::new(LanguageMode::Detect);

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert!(tokens.is_empty());
    }

    #[test]
    #[cfg(feature = "language_detection")]
    fn it_detects_english() {
        let tokens_detected = tokenize_recipes("recipes_en.csv", LanguageMode::Detect);
        let tokens_en = tokenize_recipes("recipes_en.csv", LanguageMode::Fixed(Language::English));

        assert_eq!(tokens_detected, tokens_en);
    }

    #[test]
    #[cfg(feature = "language_detection")]
    fn it_detects_german() {
        let tokens_detected = tokenize_recipes("recipes_de.csv", LanguageMode::Detect);
        let token_de = tokenize_recipes("recipes_de.csv", LanguageMode::Fixed(Language::German));

        assert_eq!(tokens_detected, token_de);
    }

    #[test]
    fn it_matches_snapshot_en() {
        let tokens = tokenize_recipes("recipes_en.csv", LanguageMode::Fixed(Language::English));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(tokens);
        });
    }

    #[test]
    fn it_matches_snapshot_de() {
        let tokens = tokenize_recipes("recipes_de.csv", LanguageMode::Fixed(Language::German));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(tokens);
        });
    }

    #[test]
    fn it_does_not_convert_unicode_when_normalization_disabled() {
        let text = "étude";
        let tokenizer = DefaultTokenizer::builder()
            .language_mode(Language::French)
            .normalization(false)
            .stemming(false)
            .build();

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["étude"]);
    }

    #[test]
    fn it_does_not_remove_stopwords_when_stopwords_disabled() {
        let text = "i my myself we you have";
        let tokenizer = DefaultTokenizer::builder()
            .language_mode(Language::English)
            .stopwords(false)
            .build();

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(tokens, vec!["i", "my", "myself", "we", "you", "have"]);
    }

    #[test]
    fn it_does_not_stem_when_stemming_disabled() {
        let text = "connection connections connective connect";
        let tokenizer = DefaultTokenizer::builder()
            .language_mode(Language::English)
            .stemming(false)
            .build();

        let tokens: Vec<_> = tokenizer.tokenize(text).collect();

        assert_eq!(
            tokens,
            vec!["connection", "connections", "connective", "connect"]
        );
    }
}
