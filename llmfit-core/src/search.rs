//! Shared text-search matching.
//!
//! Every place that filters rows by a free-text query — the TUI model table,
//! the community leaderboard, the `search`/`list` CLI subcommands, the HTTP
//! API and the MCP `search_models` tool — routes through [`Matcher`] so they
//! all agree on what a query means.
//!
//! The interpretation is selectable via [`SearchMode`]. The default,
//! [`SearchMode::Smart`], is a strict superset of the historical
//! substring-only behaviour: anything that matched before still matches, plus
//! partial/abbreviated matches.

use regex::{Regex, RegexBuilder};
use std::fmt;
use std::str::FromStr;

/// Ceiling on the compiled size of a user-supplied pattern. The `regex` crate
/// has no backtracking so a pattern cannot blow up at match time, but a
/// pathological one can still allocate a large automaton at compile time.
const REGEX_SIZE_LIMIT: usize = 1 << 20;

/// How a free-text search query is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum SearchMode {
    /// Space-separated terms; every term must match as a substring **or** as an
    /// in-order subsequence. Superset of [`SearchMode::Substring`], so no query
    /// that used to return a row stops returning it. Default.
    #[default]
    Smart,
    /// Space-separated terms; every term must appear verbatim
    /// (case-insensitive). The historical behaviour.
    Substring,
    /// Space-separated terms; every term must appear as an in-order
    /// subsequence, e.g. `qw3c` matches `Qwen3-Coder`.
    Fuzzy,
    /// The whole query is one case-insensitive regular expression.
    Regex,
}

impl SearchMode {
    /// Every mode, in cycle order.
    pub const ALL: [SearchMode; 4] = [
        SearchMode::Smart,
        SearchMode::Substring,
        SearchMode::Fuzzy,
        SearchMode::Regex,
    ];

    /// The next mode in cycle order, wrapping around.
    pub fn next(self) -> Self {
        let idx = Self::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    /// The previous mode in cycle order, wrapping around.
    pub fn prev(self) -> Self {
        let idx = Self::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    /// Human-readable name, e.g. for the status bar.
    pub fn label(self) -> &'static str {
        match self {
            SearchMode::Smart => "smart",
            SearchMode::Substring => "substring",
            SearchMode::Fuzzy => "fuzzy",
            SearchMode::Regex => "regex",
        }
    }

    /// Compact name for the search box title, where width is scarce.
    pub fn short_label(self) -> &'static str {
        match self {
            SearchMode::Smart => "smart",
            SearchMode::Substring => "sub",
            SearchMode::Fuzzy => "fuzzy",
            SearchMode::Regex => "re",
        }
    }

    /// One-line description for help screens and `--help` output.
    pub fn description(self) -> &'static str {
        match self {
            SearchMode::Smart => "substring or subsequence per term (default)",
            SearchMode::Fuzzy => "subsequence per term, e.g. qw3c → Qwen3-Coder",
            SearchMode::Substring => "every term must appear verbatim",
            SearchMode::Regex => "whole query is a case-insensitive regex",
        }
    }
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for SearchMode {
    type Err = String;

    /// Parses a mode name. Case-insensitive, with the aliases people reach for
    /// first, so a hand-edited config or `LLMFIT_SEARCH_MODE` rarely fails.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "smart" | "auto" | "default" => Ok(SearchMode::Smart),
            "substring" | "sub" | "contains" | "exact" | "literal" | "plain" => {
                Ok(SearchMode::Substring)
            }
            "fuzzy" | "fzf" | "subsequence" => Ok(SearchMode::Fuzzy),
            "regex" | "re" | "regexp" | "regular-expression" => Ok(SearchMode::Regex),
            other => Err(format!(
                "unknown search mode '{other}' (expected one of: smart, substring, fuzzy, regex)"
            )),
        }
    }
}

/// How far apart the characters of a fuzzy (subsequence) match may be spread.
///
/// Expressed as a multiple of the query term's length: `MaxSpan(3)` accepts a
/// match whose tightest window is at most three times as long as the term, so
/// `gm7` still finds `gemma-7b` (window 7, term 3) but stops matching every
/// long model id that happens to contain g, m and 7 in that order.
///
/// A contiguous substring always has a window equal to the term length, so no
/// tightness setting can ever reject something [`SearchMode::Substring`] would
/// have matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fuzz {
    /// Any gap is acceptable — a plain subsequence test.
    Unlimited,
    /// Reject matches whose tightest window exceeds `n x term length`.
    MaxSpan(u8),
}

impl Default for Fuzz {
    /// Measured against the bundled 10k-model catalog: `3` keeps every
    /// intended abbreviation (`qw3c` -> Qwen3-Coder, `gm7` -> gemma-7b,
    /// `llm8` -> llama-3.1-8b) while cutting incidental hits roughly fivefold.
    /// `2` is tighter but already drops `gm7` -> `gemma-7b`.
    fn default() -> Self {
        Fuzz::MaxSpan(3)
    }
}

impl Fuzz {
    /// Tightest allowed window for a term of `term_len` chars, or `None` when
    /// unlimited.
    pub fn max_span(self, term_len: usize) -> Option<usize> {
        match self {
            Fuzz::Unlimited => None,
            Fuzz::MaxSpan(n) => Some(term_len.saturating_mul(n as usize).max(term_len)),
        }
    }

    /// The tightness multiplier, or `None` when unlimited.
    pub fn multiplier(self) -> Option<u8> {
        match self {
            Fuzz::Unlimited => None,
            Fuzz::MaxSpan(n) => Some(n),
        }
    }

    /// Smallest and largest selectable multipliers. `1` is contiguous-only.
    pub const MIN: u8 = 1;
    pub const MAX: u8 = 9;

    /// The tightness values the TUI cycles through with a single key.
    ///
    /// A subset of the `1..=MAX` range that `--fuzz` accepts: enough spread to
    /// be useful, few enough that cycling stays quick.
    pub const LADDER: [Fuzz; 5] = [
        Fuzz::MaxSpan(1),
        Fuzz::MaxSpan(2),
        Fuzz::MaxSpan(3),
        Fuzz::MaxSpan(5),
        Fuzz::Unlimited,
    ];

    /// Next rung of [`Fuzz::LADDER`], wrapping. An arbitrary `--fuzz` value
    /// not on the ladder snaps to the next rung above it.
    pub fn cycle(self) -> Self {
        match Self::LADDER.iter().position(|f| *f == self) {
            Some(i) => Self::LADDER[(i + 1) % Self::LADDER.len()],
            None => Self::LADDER
                .iter()
                .find(|f| match (f, self) {
                    (Fuzz::Unlimited, _) => true,
                    (Fuzz::MaxSpan(a), Fuzz::MaxSpan(b)) => *a > b,
                    _ => false,
                })
                .copied()
                .unwrap_or(Fuzz::LADDER[0]),
        }
    }

    /// One step looser; past [`Fuzz::MAX`] this becomes [`Fuzz::Unlimited`].
    pub fn looser(self) -> Self {
        match self {
            Fuzz::Unlimited => Fuzz::Unlimited,
            Fuzz::MaxSpan(n) if n >= Self::MAX => Fuzz::Unlimited,
            Fuzz::MaxSpan(n) => Fuzz::MaxSpan(n + 1),
        }
    }

    /// One step tighter; [`Fuzz::Unlimited`] steps down to [`Fuzz::MAX`].
    pub fn tighter(self) -> Self {
        match self {
            Fuzz::Unlimited => Fuzz::MaxSpan(Self::MAX),
            Fuzz::MaxSpan(n) if n <= Self::MIN => Fuzz::MaxSpan(Self::MIN),
            Fuzz::MaxSpan(n) => Fuzz::MaxSpan(n - 1),
        }
    }

    /// Label for the search box and status bar.
    pub fn label(self) -> String {
        match self {
            Fuzz::Unlimited => "off".to_string(),
            Fuzz::MaxSpan(n) => n.to_string(),
        }
    }
}

impl FromStr for Fuzz {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let raw = s.trim().to_lowercase();
        if matches!(raw.as_str(), "off" | "none" | "unlimited" | "0") {
            return Ok(Fuzz::Unlimited);
        }
        match raw.parse::<u8>() {
            Ok(n) if (Self::MIN..=Self::MAX).contains(&n) => Ok(Fuzz::MaxSpan(n)),
            _ => Err(format!(
                "invalid fuzz '{s}' (expected {}-{} or 'off')",
                Self::MIN,
                Self::MAX
            )),
        }
    }
}

impl fmt::Display for Fuzz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

/// A query compiled once and then tested against many rows.
///
/// Construct with [`Matcher::new`] outside the row loop — in [`SearchMode::Regex`]
/// that hoists pattern compilation out of the per-row path.
#[derive(Debug, Clone)]
pub struct Matcher {
    mode: SearchMode,
    fuzz: Fuzz,
    /// Lowercased whitespace-separated terms, as chars so window arithmetic is
    /// in characters rather than bytes. Always empty in regex mode.
    terms: Vec<Vec<char>>,
    regex: Option<Regex>,
    error: Option<String>,
}

impl Matcher {
    /// Compiles `query` under `mode`.
    ///
    /// An invalid regex never fails construction and never hides rows: the
    /// matcher stops filtering entirely and records the reason in
    /// [`Matcher::error`]. Most prefixes of a real pattern are themselves
    /// invalid (`^(qwen|` while typing `^(qwen|llama)`), so the alternative —
    /// applying *some* other interpretation — would blank the table on nearly
    /// every keystroke and read as "search broke". Callers must surface
    /// [`Matcher::error`] so the unfiltered result is explained.
    pub fn new(query: &str, mode: SearchMode) -> Self {
        Self::with_fuzz(query, mode, Fuzz::default())
    }

    /// Compiles `query` under `mode` with an explicit fuzzy-match tightness.
    /// `fuzz` is ignored by [`SearchMode::Substring`] and [`SearchMode::Regex`],
    /// which have no gapped matching to loosen.
    pub fn with_fuzz(query: &str, mode: SearchMode, fuzz: Fuzz) -> Self {
        let blank = Self {
            mode,
            fuzz,
            terms: Vec::new(),
            regex: None,
            error: None,
        };

        if mode == SearchMode::Regex {
            // Compile the query verbatim. Whitespace is significant in a
            // pattern — ` 7b` and `7b` are different regexes, and a trailing
            // space can be the second half of an escape — so trimming here
            // would silently compile something other than what was typed.
            // Only a truly empty query means "no filter".
            if query.is_empty() {
                return blank;
            }
            return match RegexBuilder::new(query)
                .case_insensitive(true)
                .size_limit(REGEX_SIZE_LIMIT)
                .dfa_size_limit(REGEX_SIZE_LIMIT)
                .build()
            {
                Ok(regex) => Self {
                    mode,
                    fuzz,
                    terms: Vec::new(),
                    regex: Some(regex),
                    error: None,
                },
                Err(err) => Self {
                    mode,
                    fuzz,
                    terms: Vec::new(),
                    regex: None,
                    error: Some(compact_regex_error(&err)),
                },
            };
        }

        // Term modes split on whitespace, so surrounding whitespace is noise.
        let terms = lowercase_terms(query);
        if terms.is_empty() {
            return blank;
        }
        Self {
            mode,
            fuzz,
            terms,
            regex: None,
            error: None,
        }
    }

    /// The mode this matcher was built for.
    pub fn mode(&self) -> SearchMode {
        self.mode
    }

    /// Why a [`SearchMode::Regex`] pattern was rejected. When this is set the
    /// matcher does no filtering at all — see [`Matcher::new`].
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// True when this matcher accepts every row, either because the query was
    /// blank or because the pattern did not compile.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.regex.is_none()
    }

    /// How loose a subsequence match may be.
    pub fn fuzz(&self) -> Fuzz {
        self.fuzz
    }

    /// Tests a row whose searchable text is spread over several fields.
    ///
    /// **Every term must match, and a term matches if it matches any single
    /// field** — AND across terms, OR across fields. So `llama meta` still
    /// finds a Llama model whose provider is Meta, while a term is never
    /// satisfied by text that only exists by straddling two fields.
    ///
    /// Matching per field rather than against the fields concatenated is what
    /// makes the non-substring modes usable: `^gemma` and `-7b$` anchor to a
    /// field instead of to the whole blob, and a subsequence like `qw3c` can
    /// no longer be assembled from characters scavenged out of six unrelated
    /// columns.
    pub fn matches_fields<'a, I>(&self, fields: I) -> bool
    where
        I: IntoIterator<Item = &'a str>,
    {
        if self.is_empty() {
            return true;
        }
        if let Some(regex) = &self.regex {
            return fields.into_iter().any(|field| regex.is_match(field));
        }
        let lowered = lower_fields(fields);
        self.terms
            .iter()
            .all(|term| lowered.iter().any(|field| self.term_hit(field, term)))
    }

    /// Tests a row whose searchable text is a single string.
    /// Equivalent to [`Matcher::matches_fields`] with one field.
    pub fn matches(&self, haystack: &str) -> bool {
        self.matches_fields([haystack])
    }

    /// Relevance score for a matching row, or `None` if it does not match.
    ///
    /// Higher is better. Only meaningful relative to other rows scored by the
    /// same matcher. Fields are scored in the order given, and earlier fields
    /// count for more — callers should pass the most identifying field (the
    /// model name) first.
    ///
    /// Computing a score costs more than [`Matcher::matches_fields`], so this
    /// is called only when results are actually being ranked by relevance.
    pub fn score_fields<'a, I>(&self, fields: I) -> Option<i32>
    where
        I: IntoIterator<Item = &'a str>,
    {
        if self.is_empty() {
            return Some(0);
        }

        if let Some(regex) = &self.regex {
            // A regex either matches or not; rank by which field matched and
            // how early, so name hits outrank a hit buried in a GGUF repo path.
            let mut best: Option<i32> = None;
            for (idx, field) in fields.into_iter().enumerate() {
                if let Some(m) = regex.find(field) {
                    let start =
                        i32::try_from(field[..m.start()].chars().count()).unwrap_or(i32::MAX);
                    let cand = FIELD_WEIGHT
                        .saturating_sub(field_penalty(idx))
                        .saturating_sub(start);
                    best = Some(best.map_or(cand, |b: i32| b.max(cand)));
                }
            }
            return best;
        }

        let lowered = lower_fields(fields);
        let mut total = 0i32;
        for term in &self.terms {
            let mut best: Option<i32> = None;
            for (idx, field) in lowered.iter().enumerate() {
                if let Some(points) = self.term_score(field, term) {
                    let cand = points.saturating_sub(field_penalty(idx));
                    best = Some(best.map_or(cand, |b: i32| b.max(cand)));
                }
            }
            // Every term must match somewhere, same rule as matches_fields.
            // Saturating: a pathological number of terms or fields must
            // degrade the ordering, never wrap it (which in a release build
            // would silently invert it) or panic in a debug build.
            total = total.saturating_add(best?);
        }
        Some(total)
    }

    /// Whether `term` matches `field` under the current mode and fuzz setting.
    fn term_hit(&self, field: &[char], term: &[char]) -> bool {
        match self.mode {
            SearchMode::Substring => contains_slice(field, term).is_some(),
            SearchMode::Fuzzy => self.subsequence_hit(field, term),
            SearchMode::Smart => {
                contains_slice(field, term).is_some() || self.subsequence_hit(field, term)
            }
            // Unreachable: a compiled regex takes the branch above, and a
            // failed one leaves `terms` empty so `is_empty()` returns early.
            SearchMode::Regex => true,
        }
    }

    /// A subsequence match that also satisfies the fuzz tightness limit.
    fn subsequence_hit(&self, field: &[char], term: &[char]) -> bool {
        match self.fuzz.max_span(term.len()) {
            None => is_subsequence(field, term),
            Some(limit) => min_window(field, term).is_some_and(|span| span <= limit),
        }
    }

    /// Points for one term against one field, or `None` if it does not match.
    ///
    /// The ladder is deliberately coarse and gap-free so the ordering is easy
    /// to reason about: whole-field equality beats a prefix, a prefix beats a
    /// substring elsewhere, and any substring beats a gapped subsequence.
    fn term_score(&self, field: &[char], term: &[char]) -> Option<i32> {
        if self.mode != SearchMode::Fuzzy {
            if field == term {
                return Some(1000);
            }
            if let Some(at) = contains_slice(field, term) {
                let base: i32 = if at == 0 { 700 } else { 500 };
                // Prefer earlier hits, and hits that cover more of the field.
                let coverage = (term.len() * 100 / field.len().max(1)).min(100) as i32;
                return Some(base - at.min(50) as i32 + coverage);
            }
            if self.mode == SearchMode::Substring {
                return None;
            }
        } else if field == term {
            return Some(1000);
        } else if let Some(at) = contains_slice(field, term) {
            // Contiguous is the tightest possible subsequence; rank it top.
            let coverage = (term.len() * 100 / field.len().max(1)).min(100) as i32;
            let base: i32 = if at == 0 { 700 } else { 500 };
            return Some(base - at.min(50) as i32 + coverage);
        }

        // Gapped subsequence: tighter is better.
        let span = min_window(field, term)?;
        if let Some(limit) = self.fuzz.max_span(term.len())
            && span > limit
        {
            return None;
        }
        let slack = i32::try_from(span - term.len()).unwrap_or(i32::MAX);
        Some(200 - slack.min(150))
    }
}

/// Weight given to a regex hit before per-field and position penalties.
const FIELD_WEIGHT: i32 = 1000;
/// How much each field position costs, so a name hit outranks a later field.
const FIELD_PENALTY: i32 = 40;

/// Penalty for a field at position `idx`, clamped so an absurd field count
/// cannot overflow the score.
fn field_penalty(idx: usize) -> i32 {
    i32::try_from(idx)
        .unwrap_or(i32::MAX)
        .saturating_mul(FIELD_PENALTY)
}

fn lower_fields<'a, I>(fields: I) -> Vec<Vec<char>>
where
    I: IntoIterator<Item = &'a str>,
{
    fields
        .into_iter()
        .map(|f| f.chars().flat_map(char::to_lowercase).collect())
        .collect()
}

/// Index of the first contiguous occurrence of `needle` in `haystack`.
fn contains_slice(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Width of the tightest window of `haystack` that contains `needle` as an
/// in-order subsequence, or `None` if it does not occur at all.
///
/// Standard minimum-window-subsequence scan: walk forward to the end of a
/// match, then walk back from that end to pull the start as far right as
/// possible, then restart just after that start. Each window is found in two
/// linear passes, and fields here are short.
fn min_window(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    let mut best: Option<usize> = None;
    let mut from = 0usize;
    while from < haystack.len() {
        // Forward: leftmost end of a match starting at or after `from`.
        let mut k = 0usize;
        let mut end = None;
        for (i, &c) in haystack.iter().enumerate().skip(from) {
            if c == needle[k] {
                k += 1;
                if k == needle.len() {
                    end = Some(i);
                    break;
                }
            }
        }
        let Some(end) = end else { break };

        // Backward: tighten the start for that end.
        let mut k = needle.len();
        let mut start = end;
        for i in (0..=end).rev() {
            if haystack[i] == needle[k - 1] {
                k -= 1;
                if k == 0 {
                    start = i;
                    break;
                }
            }
        }
        let width = end - start + 1;
        best = Some(best.map_or(width, |b: usize| b.min(width)));
        if width == needle.len() {
            break; // contiguous; cannot do better
        }
        from = start + 1;
    }
    best
}

fn lowercase_terms(query: &str) -> Vec<Vec<char>> {
    query
        .to_lowercase()
        .split_whitespace()
        .map(|term| term.chars().collect())
        .collect()
}

/// `regex::Error` renders as a multi-line diagram. Status bars have one line.
fn compact_regex_error(err: &regex::Error) -> String {
    let text = err.to_string();
    text.lines()
        .map(str::trim)
        .rfind(|line| {
            !line.is_empty() && !line.starts_with('^') && !line.starts_with("regex parse error")
        })
        .filter(|line| !line.is_empty())
        .unwrap_or("invalid pattern")
        .to_string()
}

/// True when every char of `needle` appears in `haystack` in order (not
/// necessarily contiguously). Both are expected to be lowercased already.
fn is_subsequence(haystack: &[char], needle: &[char]) -> bool {
    let mut hay = haystack.iter();
    // `Iterator::any` consumes up to and including the match, so each needle
    // char resumes scanning after the previous one — an in-order walk.
    needle.iter().all(|nc| hay.any(|hc| hc == nc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_matches_everything() {
        for mode in SearchMode::ALL {
            let m = Matcher::new("", mode);
            assert!(m.is_empty(), "{mode} should be empty");
            assert!(m.matches("anything"));
        }
    }

    #[test]
    fn substring_mode_requires_every_term_verbatim() {
        let m = Matcher::new("llama meta", SearchMode::Substring);
        assert!(m.matches("Llama-3.1-8B Meta"));
        assert!(!m.matches("Llama-3.1-8B Mistral"));
    }

    #[test]
    fn substring_mode_rejects_gapped_terms() {
        let m = Matcher::new("qw3c", SearchMode::Substring);
        assert!(!m.matches("Qwen3-Coder"));
    }

    #[test]
    fn fuzzy_mode_accepts_gapped_terms() {
        let m = Matcher::new("qw3c", SearchMode::Fuzzy);
        assert!(m.matches("Qwen3-Coder"));
        assert!(!m.matches("Llama-3.1-8B"));
    }

    #[test]
    fn fuzzy_mode_requires_order() {
        // Unlimited fuzz so this tests ordering alone: at the default
        // tightness `cq` is too spread out for either candidate.
        let m = Matcher::with_fuzz("cq", SearchMode::Fuzzy, Fuzz::Unlimited);
        assert!(!m.matches("Qwen-Coder"), "reversed order must not match");
        assert!(m.matches("Coder-Qwen"));
    }

    #[test]
    fn smart_mode_is_a_superset_of_substring() {
        let substring = Matcher::new("llama", SearchMode::Substring);
        let smart = Matcher::new("llama", SearchMode::Smart);
        let haystack = "Llama-3.1-8B Meta";
        assert!(substring.matches(haystack));
        assert!(smart.matches(haystack));

        let smart_gapped = Matcher::new("lma", SearchMode::Smart);
        assert!(smart_gapped.matches(haystack));
        assert!(!Matcher::new("lma", SearchMode::Substring).matches(haystack));
    }

    #[test]
    fn matching_is_case_insensitive_in_every_mode() {
        for mode in SearchMode::ALL {
            let m = Matcher::new("LLAMA", mode);
            assert!(m.matches("llama-3.1-8b"), "{mode} lost case-insensitivity");
        }
    }

    #[test]
    fn regex_mode_matches_patterns() {
        let m = Matcher::new(r"llama-3\.[12]", SearchMode::Regex);
        assert!(m.error().is_none());
        assert!(m.matches("Llama-3.1-8B"));
        assert!(m.matches("llama-3.2-3b"));
        assert!(!m.matches("Llama-2-7B"));
    }

    #[test]
    fn regex_mode_supports_alternation_and_anchors() {
        let m = Matcher::new("^(qwen|llama)", SearchMode::Regex);
        assert!(m.matches("qwen3-coder"));
        assert!(!m.matches("meta llama"));
    }

    #[test]
    fn regex_mode_does_not_split_on_whitespace() {
        // A space is a literal in regex mode, not a term separator.
        let m = Matcher::new("llama meta", SearchMode::Regex);
        assert!(m.matches("llama meta"));
        assert!(!m.matches("meta llama"));
    }

    #[test]
    fn invalid_regex_stops_filtering_and_reports_why() {
        let m = Matcher::new("qwen(", SearchMode::Regex);
        assert!(m.error().is_some(), "must explain the rejection");
        // Half-typed patterns must not hide rows; the error is what tells the
        // user the filter is not being applied.
        assert!(m.is_empty());
        assert!(m.matches("Qwen3-Coder"));
        assert!(m.matches("Llama-3.1-8B"));
    }

    #[test]
    fn every_prefix_of_a_valid_pattern_is_safe_to_type() {
        // Typing "^(qwen|llama)" one key at a time must never panic and never
        // produce a matcher that silently hides everything.
        let target = "^(qwen|llama)";
        for end in 1..=target.len() {
            let m = Matcher::new(&target[..end], SearchMode::Regex);
            if m.error().is_some() {
                assert!(m.is_empty(), "prefix {:?} filtered anyway", &target[..end]);
            }
            // Either way this must not panic.
            let _ = m.matches("qwen3-coder");
        }
        let complete = Matcher::new(target, SearchMode::Regex);
        assert!(complete.error().is_none());
        assert!(complete.matches("qwen3-coder"));
    }

    #[test]
    fn regex_mode_preserves_significant_leading_and_trailing_whitespace() {
        // ` 7b` is a different pattern from `7b`: it requires a literal space.
        let m = Matcher::new(" 7b", SearchMode::Regex);
        assert!(m.error().is_none());
        assert!(m.matches("Llama 7b"));
        assert!(!m.matches("Llama-7b"), "the leading space must be honoured");

        // Trimming a trailing space would leave a dangling escape.
        let escaped = Matcher::new(r"foo\ ", SearchMode::Regex);
        assert!(escaped.error().is_none());
        assert!(escaped.matches("foo bar"));
    }

    #[test]
    fn term_modes_ignore_surrounding_whitespace() {
        for mode in [SearchMode::Smart, SearchMode::Substring, SearchMode::Fuzzy] {
            let m = Matcher::new("  llama   8b  ", mode);
            assert!(m.matches_fields(["Llama-3.1", "8B"]), "{mode}");
            assert!(!m.matches_fields(["Mistral", "8B"]), "{mode}");
        }
        // Whitespace-only is still "no query" for term modes.
        assert!(Matcher::new("   ", SearchMode::Smart).is_empty());
    }

    #[test]
    fn a_whitespace_only_regex_is_a_real_pattern() {
        // Unlike the term modes, a lone space is a valid pattern that means
        // "contains a space", so it must not be treated as an empty query.
        let m = Matcher::new(" ", SearchMode::Regex);
        assert!(!m.is_empty());
        assert!(m.matches("Llama 3"));
        assert!(!m.matches("Llama-3"));
    }

    #[test]
    fn invalid_regex_error_is_one_line() {
        let m = Matcher::new("[a-", SearchMode::Regex);
        let err = m.error().expect("invalid class should error");
        assert!(!err.contains('\n'), "status bars render one line: {err:?}");
        assert!(!err.is_empty());
    }

    #[test]
    fn terms_are_anded_but_may_land_in_different_fields() {
        let m = Matcher::new("llama meta", SearchMode::Substring);
        assert!(m.matches_fields(["Llama-3.1-8B", "Meta", "8B"]));
        assert!(!m.matches_fields(["Llama-3.1-8B", "Mistral", "8B"]));
    }

    #[test]
    fn a_term_is_never_satisfied_by_straddling_two_fields() {
        // "8bmeta" must not be manufactured by concatenating adjacent fields.
        let m = Matcher::new("8bmeta", SearchMode::Substring);
        assert!(!m.matches_fields(["Llama-3.1-8B", "Meta"]));
        // Split into two terms it matches, because each term lands in a field.
        let spaced = Matcher::new("8b meta", SearchMode::Substring);
        assert!(spaced.matches_fields(["Llama-3.1-8B", "Meta"]));
    }

    #[test]
    fn fuzzy_terms_cannot_be_assembled_from_scraps_of_many_fields() {
        // Against the fields joined into one blob, "gm7" is a subsequence of
        // "gemma-2b Test 7B" and would match the wrong model. Per field it
        // does not.
        let m = Matcher::new("gm7", SearchMode::Fuzzy);
        assert!(!m.matches_fields(["gemma-2b", "Test", "7B", "General"]));
        assert!(m.matches_fields(["gemma-7b", "Test", "7B", "General"]));
    }

    #[test]
    fn regex_anchors_bind_to_a_field_not_to_the_whole_row() {
        let fields = ["gemma-7b", "Google", "7B", "General"];
        assert!(Matcher::new("^gemma", SearchMode::Regex).matches_fields(fields));
        // `$` would never match if the fields were concatenated.
        assert!(Matcher::new("-7b$", SearchMode::Regex).matches_fields(fields));
        assert!(!Matcher::new("^oogle", SearchMode::Regex).matches_fields(fields));
    }

    #[test]
    fn multibyte_queries_do_not_panic() {
        for mode in SearchMode::ALL {
            let m = Matcher::new("你好 😀", mode);
            assert!(!m.matches("plain ascii"));
            let _ = m.matches("你好 世界 😀");
        }
        assert!(Matcher::new("你好", SearchMode::Fuzzy).matches("你我好"));
    }

    // --- fuzz tightness ---

    #[test]
    fn default_fuzz_keeps_intended_abbreviations() {
        // Regression net for the tuning of Fuzz::default(). Each of these is a
        // match a user would expect from typing an abbreviation.
        for (query, field) in [
            ("qw3c", "Qwen3-Coder"),
            ("gm7", "gemma-7b"),
            ("llm8", "llama-3.1-8b"),
            ("mstrl", "mistral-7b"),
        ] {
            assert!(
                Matcher::new(query, SearchMode::Fuzzy).matches(field),
                "{query} should still match {field} at default tightness"
            );
        }
    }

    #[test]
    fn tighter_fuzz_rejects_looser_matches() {
        let field = "quanttrio/qwen3-coder-480b-instruct";
        // g..p..t scattered across a long id: plausible only at high slack.
        let loose = Matcher::with_fuzz("q4i", SearchMode::Fuzzy, Fuzz::Unlimited);
        assert!(loose.matches(field));
        let tight = Matcher::with_fuzz("q4i", SearchMode::Fuzzy, Fuzz::MaxSpan(2));
        assert!(!tight.matches(field), "tight fuzz should reject the sprawl");
    }

    #[test]
    fn fuzz_of_one_accepts_only_contiguous_matches() {
        let m = Matcher::with_fuzz("qwen", SearchMode::Fuzzy, Fuzz::MaxSpan(1));
        assert!(m.matches("Qwen3-Coder"), "contiguous must still match");
        assert!(!m.matches("Q-w-e-n"), "any gap must be rejected");
    }

    #[test]
    fn no_tightness_can_reject_what_substring_matches() {
        // A contiguous hit has a window equal to the term length, so it
        // survives even the tightest setting. This is what keeps Smart a
        // superset of Substring at every fuzz value.
        let field = "meta-llama/Llama-3.1-8B-Instruct";
        for n in Fuzz::MIN..=Fuzz::MAX {
            let m = Matcher::with_fuzz("llama", SearchMode::Smart, Fuzz::MaxSpan(n));
            assert!(m.matches(field), "fuzz {n} rejected a substring match");
        }
    }

    #[test]
    fn fuzz_cycle_visits_every_rung_and_wraps() {
        let mut seen = Vec::new();
        let mut f = Fuzz::LADDER[0];
        for _ in 0..Fuzz::LADDER.len() {
            seen.push(f);
            f = f.cycle();
        }
        assert_eq!(f, Fuzz::LADDER[0], "cycle must wrap");
        for rung in Fuzz::LADDER {
            assert!(seen.contains(&rung), "{rung} missing from cycle");
        }
    }

    #[test]
    fn fuzz_cycle_snaps_an_off_ladder_value_upward() {
        // --fuzz 4 is legal but not a rung; cycling moves to the next one up.
        assert_eq!(Fuzz::MaxSpan(4).cycle(), Fuzz::MaxSpan(5));
        assert_eq!(Fuzz::MaxSpan(7).cycle(), Fuzz::Unlimited);
    }

    #[test]
    fn fuzz_steps_saturate_at_both_ends() {
        assert_eq!(Fuzz::MaxSpan(Fuzz::MIN).tighter(), Fuzz::MaxSpan(Fuzz::MIN));
        assert_eq!(Fuzz::MaxSpan(Fuzz::MAX).looser(), Fuzz::Unlimited);
        assert_eq!(Fuzz::Unlimited.looser(), Fuzz::Unlimited);
        assert_eq!(Fuzz::Unlimited.tighter(), Fuzz::MaxSpan(Fuzz::MAX));
    }

    #[test]
    fn fuzz_parses_numbers_and_off() {
        assert_eq!("3".parse::<Fuzz>().unwrap(), Fuzz::MaxSpan(3));
        assert_eq!("off".parse::<Fuzz>().unwrap(), Fuzz::Unlimited);
        assert_eq!("0".parse::<Fuzz>().unwrap(), Fuzz::Unlimited);
        assert!("99".parse::<Fuzz>().is_err());
        assert!("nonsense".parse::<Fuzz>().is_err());
        for n in Fuzz::MIN..=Fuzz::MAX {
            let f = Fuzz::MaxSpan(n);
            assert_eq!(f.label().parse::<Fuzz>().unwrap(), f);
        }
        assert_eq!(
            Fuzz::Unlimited.label().parse::<Fuzz>().unwrap(),
            Fuzz::Unlimited
        );
    }

    #[test]
    fn min_window_finds_the_tightest_not_the_first() {
        let hay: Vec<char> = "a__b_ab".chars().collect();
        let needle: Vec<char> = "ab".chars().collect();
        // The leftmost match spans a(0)..b(3) = 4; the tightest is a(5)b(6) = 2.
        assert_eq!(min_window(&hay, &needle), Some(2));
        let none: Vec<char> = "ba".chars().collect();
        assert_eq!(min_window(&none, &needle), None);
    }

    // --- relevance scoring ---

    #[test]
    fn exact_and_prefix_matches_outrank_buried_ones() {
        let m = Matcher::new("llama", SearchMode::Smart);
        let exact = m.score_fields(["llama"]).unwrap();
        let prefix = m.score_fields(["llama-3.1-8b"]).unwrap();
        let buried = m.score_fields(["meta-llama-3.1-8b"]).unwrap();
        let gapped = m.score_fields(["l-l-a-m-a"]).unwrap();
        assert!(exact > prefix, "{exact} !> {prefix}");
        assert!(prefix > buried, "{prefix} !> {buried}");
        assert!(buried > gapped, "{buried} !> {gapped}");
    }

    #[test]
    fn a_name_hit_outranks_the_same_hit_in_a_later_field() {
        let m = Matcher::new("meta", SearchMode::Smart);
        let in_name = m.score_fields(["meta-llama-3", "other"]).unwrap();
        let in_provider = m.score_fields(["other", "meta-llama-3"]).unwrap();
        assert!(in_name > in_provider, "{in_name} !> {in_provider}");
    }

    #[test]
    fn tighter_fuzzy_matches_score_higher() {
        let m = Matcher::with_fuzz("qwc", SearchMode::Fuzzy, Fuzz::Unlimited);
        let tight = m.score_fields(["qwc"]).unwrap();
        let loose = m.score_fields(["q_w_______________c"]).unwrap();
        assert!(tight > loose, "{tight} !> {loose}");
    }

    #[test]
    fn score_agrees_with_matches_on_whether_a_row_qualifies() {
        let rows = [
            vec!["Qwen3-Coder", "Alibaba", "30B"],
            vec!["Llama-3.1-8B", "Meta", "8B"],
            vec!["gemma-7b", "Google", "7B"],
        ];
        for mode in SearchMode::ALL {
            for query in ["qwen", "qw3c", "^gem", "meta 8b", "zzz"] {
                let m = Matcher::new(query, mode);
                for row in &rows {
                    assert_eq!(
                        m.matches_fields(row.iter().copied()),
                        m.score_fields(row.iter().copied()).is_some(),
                        "{mode}/{query} disagreed on {row:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn scoring_an_empty_query_is_flat() {
        let m = Matcher::new("", SearchMode::Smart);
        assert_eq!(m.score_fields(["anything"]), Some(0));
        assert_eq!(m.score_fields(["other"]), Some(0));
    }

    #[test]
    fn invalid_regex_scores_every_row_equally() {
        let m = Matcher::new("qwen(", SearchMode::Regex);
        assert!(m.error().is_some());
        assert_eq!(m.score_fields(["Qwen3"]), Some(0));
        assert_eq!(m.score_fields(["Llama"]), Some(0));
    }

    #[test]
    fn mode_cycles_through_every_variant_and_wraps() {
        let mut seen = Vec::new();
        let mut mode = SearchMode::default();
        for _ in 0..SearchMode::ALL.len() {
            seen.push(mode);
            mode = mode.next();
        }
        assert_eq!(mode, SearchMode::default(), "cycle must wrap");
        for variant in SearchMode::ALL {
            assert!(seen.contains(&variant), "{variant} missing from cycle");
        }
    }

    #[test]
    fn prev_is_the_inverse_of_next() {
        for mode in SearchMode::ALL {
            assert_eq!(mode.next().prev(), mode);
            assert_eq!(mode.prev().next(), mode);
        }
        assert_eq!(
            SearchMode::Smart.prev(),
            SearchMode::Regex,
            "must wrap back"
        );
    }

    #[test]
    fn mode_parses_names_and_aliases_case_insensitively() {
        assert_eq!("Smart".parse::<SearchMode>().unwrap(), SearchMode::Smart);
        assert_eq!("RE".parse::<SearchMode>().unwrap(), SearchMode::Regex);
        assert_eq!(" fzf ".parse::<SearchMode>().unwrap(), SearchMode::Fuzzy);
        assert_eq!(
            "contains".parse::<SearchMode>().unwrap(),
            SearchMode::Substring
        );
        assert!("nonsense".parse::<SearchMode>().is_err());
    }

    #[test]
    fn mode_labels_round_trip_through_parsing() {
        for mode in SearchMode::ALL {
            assert_eq!(mode.label().parse::<SearchMode>().unwrap(), mode);
            assert_eq!(mode.short_label().parse::<SearchMode>().unwrap(), mode);
        }
    }

    #[test]
    fn classic_redos_pattern_stays_linear() {
        // `(a+)+$` against a long run of 'a' followed by a non-match is the
        // textbook catastrophic-backtracking input. The regex crate has no
        // backtracking, so this returns immediately; the size limit is what
        // keeps *compilation* bounded.
        let m = Matcher::new("(a+)+$", SearchMode::Regex);
        assert!(m.error().is_none());
        assert!(!m.matches(&format!("{}b", "a".repeat(5000))));
        assert!(m.matches(&"a".repeat(5000)));
    }

    #[test]
    fn oversized_pattern_is_rejected_rather_than_compiled() {
        // Bounded counted repetition is the cheapest way to blow the compiled
        // size limit; it must surface as an error, not an allocation spike.
        let m = Matcher::new(&format!("a{{{}}}", 1_000_000), SearchMode::Regex);
        assert!(m.error().is_some(), "oversized pattern must be rejected");
    }

    // --- property-style checks ---
    //
    // These found a real integer-overflow in `score_fields`; they stay.

    /// Deterministic PRNG so a failure is always reproducible.
    struct Xorshift(u64);
    impl Xorshift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn range(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
    }

    fn rand_string(rng: &mut Xorshift, alphabet: &[char], max_len: usize) -> String {
        let len = rng.range(max_len + 1);
        (0..len)
            .map(|_| alphabet[rng.range(alphabet.len())])
            .collect()
    }

    #[test]
    fn property_score_and_matches_always_agree_on_term_modes() {
        // Tiny overlapping alphabet maximises partial matches, gapped
        // subsequences and shared prefixes.
        let alphabet: Vec<char> = "ab3-_ x".chars().collect();
        let mut rng = Xorshift(0x9E3779B97F4A7C15);
        for _ in 0..4000 {
            let query = rand_string(&mut rng, &alphabet, 6);
            let field = rand_string(&mut rng, &alphabet, 14);
            for mode in [SearchMode::Smart, SearchMode::Substring, SearchMode::Fuzzy] {
                for fuzz in [Fuzz::MaxSpan(1), Fuzz::MaxSpan(3), Fuzz::Unlimited] {
                    let m = Matcher::with_fuzz(&query, mode, fuzz);
                    assert_eq!(
                        m.matches(&field),
                        m.score_fields([field.as_str()]).is_some(),
                        "{mode}/{fuzz} disagreed on query={query:?} field={field:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn property_score_and_matches_agree_across_multiple_fields() {
        let alphabet: Vec<char> = "ab3-_ x".chars().collect();
        let mut rng = Xorshift(0xDEADBEEFCAFEF00D);
        for _ in 0..2000 {
            let query = rand_string(&mut rng, &alphabet, 8);
            let fields: Vec<String> = (0..4)
                .map(|_| rand_string(&mut rng, &alphabet, 10))
                .collect();
            let refs: Vec<&str> = fields.iter().map(String::as_str).collect();
            for mode in [SearchMode::Smart, SearchMode::Substring, SearchMode::Fuzzy] {
                let m = Matcher::new(&query, mode);
                assert_eq!(
                    m.matches_fields(refs.iter().copied()),
                    m.score_fields(refs.iter().copied()).is_some(),
                    "{mode} disagreed on query={query:?} fields={fields:?}"
                );
            }
        }
    }

    #[test]
    fn property_score_and_matches_agree_in_regex_mode() {
        let patterns = [
            "a", "ab", "a+", "a*b", "^a", "b$", "a|b", "[ab]+", "a{1,2}", "3", "-", "a-b", "^$",
            ".*", "a.b", "(a|b)+", "x?y",
        ];
        let alphabet: Vec<char> = "ab3-_ xy".chars().collect();
        let mut rng = Xorshift(0xD1B54A32D192ED03);
        for pattern in patterns {
            let m = Matcher::new(pattern, SearchMode::Regex);
            for _ in 0..200 {
                let fields: Vec<String> = (0..3)
                    .map(|_| rand_string(&mut rng, &alphabet, 10))
                    .collect();
                let refs: Vec<&str> = fields.iter().map(String::as_str).collect();
                assert_eq!(
                    m.matches_fields(refs.iter().copied()),
                    m.score_fields(refs.iter().copied()).is_some(),
                    "pattern={pattern:?} fields={fields:?}"
                );
            }
        }
    }

    #[test]
    fn scoring_saturates_instead_of_overflowing() {
        // Regression: `total += best?` and `idx * FIELD_PENALTY` used to
        // overflow i32 — a debug panic, and in release a silent wrap that
        // would invert the ranking.
        let long = "x".repeat(5000) + "needle" + &"y".repeat(5000);
        for mode in [SearchMode::Smart, SearchMode::Substring, SearchMode::Fuzzy] {
            let m = Matcher::new("needle", mode);
            assert_eq!(m.matches(&long), m.score_fields([long.as_str()]).is_some());
        }

        // Enough fields that the position penalty alone would overflow.
        let fields: Vec<String> = (0..2000).map(|i| format!("field{i}")).collect();
        let mut refs: Vec<&str> = fields.iter().map(String::as_str).collect();
        refs.push("needle-at-the-end");
        let m = Matcher::new("needle", SearchMode::Smart);
        assert_eq!(
            m.matches_fields(refs.iter().copied()),
            m.score_fields(refs.iter().copied()).is_some()
        );

        // Enough terms that the accumulation would overflow.
        let query = vec!["a"; 200_000].join(" ");
        let m = Matcher::new(&query, SearchMode::Smart);
        assert_eq!(m.matches("a"), m.score_fields(["a"]).is_some());
    }
}
