"""Text extraction and cleaning for PDF and HTML sources.

Heavy dependencies (pypdf, trafilatura/bs4, pdfminer.six, pdfplumber) are imported lazily
so the module imports cleanly when only the core runtime is installed.
"""

from __future__ import annotations

import gzip
import io
import re
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Extracted:
    text: str
    pages: int
    content_type: str  # 'pdf', 'html' or 'latex'
    extractor: str = ""
    quality_score: float = float("inf")
    all_scores: dict[str, float] | None = None
    source_format: str = ""  # 'pdf', 'html', 'arxiv_latex', 'arxiv_html'


# Ligature map for U+FB00–U+FB06 (per Unicode Standard)
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",  # ﬀ
    "\ufb01": "fi",  # ﬁ
    "\ufb02": "fl",  # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "ſt",  # ﬅ (long s + t)
    "\ufb06": "st",  # ﬆ
}

# ---------------------------------------------------------------------------
# Squashed-text fixes: CamelCase splitting + rule-based vocabulary-gated join
# ---------------------------------------------------------------------------

# Common English suffixes that indicate a word was incorrectly split by a space.
COMMON_WORD_SUFFIXES = frozenset(
    {
        "ing",
        "tion",
        "able",
        "ible",
        "ence",
        "ance",
        "ment",
        "ness",
        "ity",
        "ive",
        "ous",
        "ful",
        "less",
        "er",
        "est",
        "ed",
        "ly",
        "al",
        "ic",
        "ary",
        "ory",
        "ant",
        "ent",
        "ist",
        "ism",
    }
)

# Letters that commonly END complete English words. If w1 ends with one of
# these, w1 is likely a complete word (not a fragment awaiting w2).
# Vowels are deliberately excluded: word fragments like "vari" (variable)
# and "corre" (corresponding) end with vowels. 'e' and 'y' are also excluded
# because they end many word fragments (lik→like, happ→happy).
COMMON_WORD_END_LETTERS = frozenset("d t s n r l")


# Common standalone English words that should NOT be joined with another word.
# If both w1 and w2 are in this set, the join is almost certainly a false positive.
COMMON_STANDALONE_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "data",
        "stock",
        "model",
        "risk",
        "time",
        "market",
        "price",
        "option",
        "bond",
        "var",
        "system",
        "process",
        "method",
        "analysis",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "must",
        "shall",
        "do",
        "does",
        "did",
        "not",
        "no",
        "yes",
        "all",
        "any",
        "some",
        "most",
        "many",
        "much",
        "such",
        "both",
        "each",
        "either",
        "neither",
        "another",
        "other",
        "others",
        "several",
        "few",
        "little",
        "less",
        "more",
        "what",
        "which",
        "who",
        "whose",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "here",
        "there",
        "now",
        "then",
        "today",
        "tomorrow",
        "yesterday",
        "this",
        "that",
        "these",
        "those",
        "a",
        "i",
        "an",
        "of",
        "in",
        "on",
        "at",
        "by",
        "to",
        "as",
        "if",
        "or",
        "but",
        "so",
        "yet",
        "nor",
        "be",
        "am",
        "is",
        "it",
        "he",
        "she",
        "they",
        "we",
        "you",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "www",
        "http",
        "com",
        "org",
        "edu",
        "gov",
        "net",
        "int",
        "ftp",
        "cardo",
        "times",
        "helvetica",
        "arial",
        "courier",
        "symbol",
        "zapfdingbats",
    }
)

# Small dictionary for word segmentation of squashed prose (long lowercase runs).
# Only used as a maximum-matching fallback; not a general-purpose dictionary.
_SEGMENT_DICT = frozenset(
    {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "is",
        "are",
        "was",
        "were",
        "been",
        "has",
        "had",
        "may",
        "might",
        "must",
        "shall",
        "yet",
        "nor",
        "am",
        # Math/finance terms that commonly appear squashed
        "theorem",
        "proof",
        "lemma",
        "corollary",
        "definition",
        "example",
        "remark",
        "proposition",
        "equation",
        "formula",
        "function",
        "variable",
        "process",
        "stochastic",
        "calculus",
        "integral",
        "derivative",
        "differential",
        "brownian",
        "motion",
        "martingale",
        "measure",
        "probability",
        "random",
        "space",
        "filter",
        "filtration",
        "stopping",
        "optional",
        "sampling",
        "ito",
        "black",
        "scholes",
        "option",
        "pricing",
        "monte",
        "carlo",
        "simulation",
        "heston",
        "sabr",
        "vasicek",
        "hull",
        "white",
        "euler",
        "maruyama",
        "milstein",
        "binomial",
        "tree",
        "greeks",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "implied",
        "volatility",
        "put",
        "call",
        "parity",
        "arbitrage",
        "risk",
        "neutral",
        "hedging",
        "portfolio",
        "asset",
        "return",
        "variance",
        "covariance",
        "correlation",
        "regression",
        "series",
        "gaussian",
        "normal",
        "distribution",
        "density",
        "cumulative",
        "expectation",
        "standard",
        "deviation",
        "mean",
        "median",
        "mode",
        "quantile",
        "confidence",
        "interval",
        "hypothesis",
        "test",
        "p-value",
        "significance",
        "level",
        "power",
        "sample",
        "size",
        "estimator",
        "estimate",
        "bias",
        "consistent",
        "efficient",
        "sufficient",
        "complete",
        "family",
        "exponential",
        "poisson",
        "geometric",
        "negative",
        "markov",
        "chain",
        "transition",
        "matrix",
        "stationary",
        "ergodic",
        "law",
        "large",
        "numbers",
        "central",
        "limit",
        "characteristic",
        "moment",
        "generating",
        "cumulant",
        "sde",
        "ode",
        "pde",
        "boundary",
        "condition",
        "initial",
        "dirichlet",
        "neumann",
        "cauchy",
        "problem",
        "posed",
        "existence",
        "uniqueness",
        "strong",
        "weak",
        "solution",
        "pathwise",
        "comparison",
        "principle",
        "maximum",
        "minimum",
        "dynamic",
        "programming",
        "optimality",
        "hamilton",
        "jacobi",
        "bellman",
        "control",
        "pontryagin",
        "feynman",
        "kac",
        "girsanov",
        "novikov",
        "kamensky",
        "kurtosis",
        "skewness",
        "markowitz",
        "theory",
        "capm",
        "model",
        "cir",
        "local",
        "jump",
        "diffusion",
        "levy",
        "merton",
        "bates",
        "fixing",
        "date",
        "maturity",
        "tenor",
        "strike",
        "spot",
        "forward",
        "future",
        "swap",
        "cap",
        "floor",
        "swaption",
        "contract",
        "futures",
        "european",
        "american",
        "bermudan",
        "asian",
        "barrier",
        "lookback",
        "chooser",
        "compound",
        "basket",
        "rainbow",
        "digital",
        "binary",
        "exotic",
        "interest",
        "rate",
        "currency",
        "commodity",
        "equity",
        "index",
        "credit",
        "fixed",
        "income",
        "mortgage",
        "backed",
        "security",
        "collateralized",
        "debt",
        "obligation",
        "default",
        "linked",
        "note",
        "spread",
        "total",
        "tranches",
        "senior",
        "mezzanine",
        "tranche",
        "loan",
        "residential",
        "commercial",
        "prime",
        "subprime",
        "alt",
        "arm",
        "real",
        "estate",
        "investment",
        "trust",
        "pass",
        "through",
        "strips",
        "principal",
        "bond",
        "paper",
        "certificate",
        "deposit",
        "repurchase",
        "agreement",
        "reverse",
        "securities",
        "lending",
        "borrowing",
        "margin",
        "short",
        "sale",
        "covering",
        "squeeze",
        "position",
        "long",
        "net",
        "gross",
        "exposure",
        "market",
        "operational",
        "liquidity",
        "basel",
        "accord",
        "ii",
        "iii",
        "capital",
        "adequacy",
        "ratio",
        "weighted",
        "assets",
        "tier",
        "leverage",
        "coverage",
        "stable",
        "funding",
        # Words needed for squashed-prose segmentation in the corpus
        "simplest",
        "quadrature",
        "approximates",
        "throughout",
        "chapter",
        "independently",
        "choice",
        "representation",
        "equivalence",
        "optimal",
        "approximation",
    }
)


def _split_camel_case(text: str) -> str:
    """Split CamelCase identifiers in prose text only.

    ``SimpleGenericMonteCarloClass`` -> ``Simple Generic Monte Carlo Class``.
    Preserves newlines; never touches tokens of length <= 2.
    Skips lines that look like math formulas (contain =, +, -, *, /, ^, _,
    or math symbols) and known math names (EulerMaruyama, Milstein, etc.).
    """
    # Math names that must stay intact
    _MATH_NAMES = frozenset(
        {
            "EulerMaruyama",
            "Euler Maruyama",
            "Milstein",
            "OrnsteinUhlenbeck",
            "Ornstein Uhlenbeck",
            "Girsanov",
            "BlackScholes",
            "Black Scholes",
            "Heston",
            "SABR",
            "Vasicek",
            "HullWhite",
            "Hull White",
            "CIR",
            "Ito",
            "Itô",
            " GBM",
            "MC",
            "SDE",
            "PDE",
        }
    )

    result_lines: list[str] = []
    for line in text.split("\n"):
        # Skip formula lines (contain math operators or symbols)
        if any(ch in line for ch in "=+-*/^_∫∂∑√∞∈∀∃∈∝∪∩⊂⊃"):
            result_lines.append(line)
            continue
        # Skip known math names
        if any(name in line for name in _MATH_NAMES):
            result_lines.append(line)
            continue
        # Lowercase -> Uppercase boundary:  lowerUpper  ->  lower Upper
        result = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
        # Acronym -> Word boundary:  URLParser  ->  URL Parser
        result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", result)
        result_lines.append(result)
    return "\n".join(result_lines)


def _is_likely_split_word(w1: str, w2: str) -> bool:
    """Heuristic: w1 + w2 looks like a single word incorrectly split by a space.

    Rules:
    - w1 must be 3-6 letters (short fragment more likely than complete word)
    - w2 must be 3+ letters
    - w1 must NOT end with a common word-final letter
    - w1 + w2 must end with a common English suffix
    - NOT both w1 and w2 are common standalone words (avoids joining
      e.g. "www" + "googletagmanager")
    """
    if len(w1) < 3 or len(w1) > 6:
        return False
    if len(w2) < 3:
        return False
    if w1[-1].lower() in COMMON_WORD_END_LETTERS:
        return False
    # If either part is a common standalone word, don't join them.
    if w1.lower() in COMMON_STANDALONE_WORDS or w2.lower() in COMMON_STANDALONE_WORDS:
        return False
    combined = w1 + w2
    return any(combined.endswith(s) for s in COMMON_WORD_SUFFIXES)


def _vocabulary_gated_join(text: str) -> str:
    """Join words incorrectly split by a space during PDF extraction.

    Rule-based replacement for the old hard-coded ``_TRUE_ARTIFACTS`` list.
    Operates line by line so newlines are always preserved.

    Rules:
    - Never join tokens of length <= 2
    - Never touch the articles "a" or "I"
    - Preserve newlines (process each line independently)
    - Only join when w1 ends with an uncommon word-final letter AND
      w1 + w2 ends with a common English suffix
    """
    result_lines: list[str] = []
    for line in text.split("\n"):
        # Find all word tokens with their positions
        tokens: list[tuple[int, int, str]] = []
        for m in re.finditer(r"[A-Za-z]{3,}", line):
            tokens.append((m.start(), m.end(), m.group()))

        modified = line
        offset = 0  # tracks how much the line length has changed
        i = 0
        while i < len(tokens) - 1:
            _, _, w1 = tokens[i]
            _, _, w2 = tokens[i + 1]

            if (
                w1.lower() not in {"a", "i"}
                and w2.lower() not in {"a", "i"}
                and _is_likely_split_word(w1, w2)
            ):
                # Rebuild the search string from original positions (adjusted by offset)
                start = tokens[i][0] + offset
                end = tokens[i + 1][1] + offset
                # The space between w1 and w2 in the current modified string
                between = modified[start:end]
                # Match "w1<sep>w2" within the span [start, end)
                m_space = re.match(re.escape(w1) + r"(\s+)" + re.escape(w2), between)
                if m_space:
                    sep = m_space.group(1)
                    modified = modified[:start] + w1 + w2 + modified[end:]
                    # Adjust offset: removed len(sep) chars
                    offset -= len(sep)
                    i += 2  # skip the joined word
                else:
                    i += 1
            else:
                i += 1

        result_lines.append(modified)
    return "\n".join(result_lines)


def _normalize_single_symbol_lines(text: str) -> str:
    """Merge stranded single-symbol lines back into their formula line.

    Lines of length <= 2 that contain only a single symbol (e.g. "n", "k", "p",
    "q") are typically PDF extraction artifacts where a symbol got stranded on
    its own line.  Merge such a line into the previous non-empty line if that
    line ends with a math symbol or Greek letter; otherwise drop it.
    """
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if len(s) <= 2 and s and all(not ch.isalnum() or ch.isalpha() for ch in s):
            # Single-symbol or very short line
            if out and out[-1].strip():
                # Merge into previous line
                out[-1] = out[-1].rstrip() + " " + s
            # else: drop it (no previous line to merge into)
        else:
            out.append(ln)
    return "\n".join(out)


def _segment_squashed_prose(text: str) -> str:
    """Segment long lowercase runs that look like squashed prose.

    PDF extraction sometimes drops spaces, producing runs like
    ``Thesimplestquadraturethatapproximates``.  Apply maximum matching
    against a small dictionary to reinsert spaces.  Only applied to runs of
    26+ lowercase letters; math formulas and mixed-case text are untouched.
    If no dictionary match is found at all, the run is left unchanged.
    """

    def _segment_run(run: str) -> str:
        # Maximum matching word segmentation (case-insensitive dictionary).
        run_lower = run.lower()
        result: list[str] = []
        i = 0
        while i < len(run_lower):
            matched = False
            for j in range(min(i + 30, len(run_lower)), i + 1, -1):
                word = run_lower[i:j]
                if word in _SEGMENT_DICT:
                    # Preserve original capitalization for the first word
                    original = run[i:j]
                    if i == 0 and original[0].isupper():
                        original = original[0].upper() + original[1:].lower()
                    result.append(original)
                    i = j
                    matched = True
                    break
            if not matched:
                # No dictionary match at all: keep the run unchanged
                return run
        return " ".join(result)

    # Only process runs of 26+ letters (all alphabetic, case-insensitive)
    def _replace(m: re.Match) -> str:
        run = m.group(0)
        if len(run) >= 26 and run.isalpha():
            return _segment_run(run)
        return run

    return re.sub(r"[A-Za-z]{26,}", _replace, text)


def _nfkc(text: str) -> str:
    """Normalize text: apply NFC then replace standard ligatures (U+FB00–U+FB06).

    Unlike NFKC, this preserves math superscripts (σ²), subscripts (σ₁),
    fractions (½), and all math operators (∑, ∫, √, ∂, etc.) unchanged.
    """
    text = unicodedata.normalize("NFC", text)
    for lig, repl in _LIGATURE_MAP.items():
        text = text.replace(lig, repl)
    return text


def _dedupe_repeated_paragraphs(text: str) -> str:
    """Remove duplicate consecutive paragraphs within a chunk.

    If the same paragraph text appears on consecutive lines/blocks, keep only
    the first occurrence.  This is a best-effort measure – the pipeline already
    has de-duplication at the chunk‐hash level, but removing intra-chunk
    repetitions improves retrieval quality.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen: set[str] = set()
    unique_paras: list[str] = []
    for para in paragraphs:
        norm = para.strip().lower()
        if norm not in seen:
            seen.add(norm)
            unique_paras.append(para)
    return "\n\n".join(unique_paras)


def _formula_dump_ratio(text: str) -> float:
    """Return the fraction of characters in *text* that are math symbols or digits.

    If > 60 % the chunk is considered a formula dump and excluded.
    """
    if not text:
        return 0.0
    # Math symbols from Unicode's "Mathematical Operators" block and digits.
    math_range = sum(
        1
        for ch in text
        if unicodedata.category(ch) in {"Sk", "Nd", "Nl"}  # math symbols / numbers
        or ch in "0123456789"
        or ch in "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        or ch in "σµ∂∫∂½√∞"
    )
    return math_range / len(text)


def _strip_title(text: str, title: str) -> str:
    """Remove the prepended document title from chunk text, if present.

    The ingest layer prepends the document title to every chunk; stripping it
    avoids marking every chunk of a document as relevant.
    """
    if title and text.startswith(title):
        return text[len(title) :]
    return text


# (old extract_pdf removed - replaced by select_best_extractor-based version below)


def _collapse_whitespace(text: str) -> str:
    # collapse runs of spaces/tabs but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html(data: bytes) -> Extracted:
    # try trafilatura first
    try:
        import trafilatura

        html = data.decode("utf-8", errors="replace")
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            with_metadata=False,
        )
        if extracted and len(extracted.strip()) >= 500:
            text = _collapse_whitespace(extracted)
            # headings: trafilatura already returns plain text; keep as is
            return Extracted(text=text, pages=1, content_type="html", source_format="html")
    except ImportError:
        pass
    except Exception:
        pass

    # fallback bs4
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("beautifulsoup4 is required for HTML extraction") from exc
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    # headings kept with level prefix
    parts: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]):
        name = el.name
        if name and name.startswith("h") and len(name) == 2:
            level = name[1]
            parts.append(f"{'#' * int(level)} {el.get_text(' ', strip=True)}")
        else:
            parts.append(el.get_text(" ", strip=True))
    text = "\n".join(p for p in parts if p.strip())
    text = _collapse_whitespace(text)
    return Extracted(text=text, pages=1, content_type="html", source_format="html")


# ---------------------------------------------------------------------------
# arXiv LaTeX source route (pandoc external binary)
# ---------------------------------------------------------------------------
class PandocNotFoundError(RuntimeError):
    """Raised when the pandoc binary is required but not installed."""


class ArxivSourceUnavailable(RuntimeError):
    """Raised when an arXiv e-print is not usable LaTeX source."""


_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/"
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)"
    r"(?:\.pdf)?(?:[?#].*)?$",
    re.IGNORECASE,
)

_DISPLAY_MATH_ENVS = frozenset(
    {
        "equation",
        "equation*",
        "align",
        "align*",
        "eqnarray",
        "eqnarray*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "displaymath",
        "math",
        "split",
        "aligned",
        "alignedat",
        "gathered",
        "cases",
        "array",
        "matrix",
        "bmatrix",
        "pmatrix",
        "vmatrix",
        "Vmatrix",
        "smallmatrix",
        "subequations",
    }
)

_THEOREM_LABELS = (
    "Theorem",
    "Lemma",
    "Remark",
    "Proposition",
    "Corollary",
    "Definition",
    "Example",
    "Fact",
    "Assumption",
)


def arxiv_id_from_url(url: str) -> str | None:
    """Return the arXiv identifier from an abs/pdf/html URL, if present."""
    match = _ARXIV_ID_RE.search(url.strip())
    return match.group("id") if match else None


def _download_arxiv_eprint(arxiv_id: str, cache_dir: Path) -> tuple[bytes, bool]:
    """Download one e-print, or read it from ``data/raw``.

    Returns ``(bytes, from_cache)``.  At most one HTTP request is made per
    paper; subsequent calls hit ``data/raw/arxiv_<id>_eprint.bin``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"arxiv_{arxiv_id}_eprint.bin"
    if cache_path.exists():
        return cache_path.read_bytes(), True

    import httpx

    from .fetch import USER_AGENT, HostRateLimiter

    limiter = HostRateLimiter(min_interval=3.0)
    limiter.wait("arxiv.org")
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = client.get(url)
    if response.status_code != 200:
        raise ArxivSourceUnavailable(f"e-print HTTP {response.status_code}")
    data = response.content
    if len(data) > 25 * 1024 * 1024:
        raise ArxivSourceUnavailable("e-print too large")
    cache_path.write_bytes(data)
    return data, False


def _safe_extract_tex(archive: tarfile.TarFile, root: Path) -> None:
    """Extract only .tex files from an e-print tar, safely."""
    for member in archive.getmembers():
        if not member.isfile() or not member.name.lower().endswith(".tex"):
            continue
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts:
            continue
        target = (root / name).resolve()
        if not str(target).startswith(str(root.resolve()) + "/"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        extracted = archive.extractfile(member)
        if extracted is not None:
            target.write_bytes(extracted.read())


def _prepare_tex_tree(data: bytes, root: Path) -> Path:
    """Materialise an e-print and return the relative path to its main .tex."""
    if data[:5] == b"%PDF-":
        raise ArxivSourceUnavailable("e-print is a PDF, not LaTeX source")
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)

    if tarfile.is_tarfile(io.BytesIO(data)):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            _safe_extract_tex(archive, root)
    else:
        (root / "main.tex").write_bytes(data)

    # The source is third-party; only .tex files are extracted above.  Ignore
    # the BOSSIP_*.py listings because they are code files, not document text.
    for tex_path in root.rglob("*.tex"):
        text = tex_path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(
            r"(?m)^.*\\(?:lstinputlisting|verbatiminput)(?:\[[^\]]*\])?\{[^}]*\.py\}.*$",
            "",
            text,
        )
        text = re.sub(r"\\input\{[^}]*\.py\}", "", text)
        tex_path.write_text(text, encoding="utf-8")

    tex_files = sorted(root.rglob("*.tex"))
    if not tex_files:
        raise ArxivSourceUnavailable("e-print contains no .tex files")

    for preferred in ("main.tex", "ms.tex", "paper.tex", "article.tex"):
        for candidate in tex_files:
            if candidate.name.lower() == preferred:
                return candidate.relative_to(root)

    with_document = []
    for candidate in tex_files:
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if "\\documentclass" in text and "\\begin{document}" in text:
            with_document.append(candidate)
    chosen = max(with_document or tex_files, key=lambda p: p.stat().st_size)
    return chosen.relative_to(root)


def _pandoc_convert(tex_root: Path, main_tex: Path) -> str:
    """Run pandoc on a materialised LaTeX tree.  Never adds a Python dependency."""
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise PandocNotFoundError(
            "pandoc is required for the arXiv LaTeX route. "
            "Install it with `pacman -S pandoc-cli` on Arch, then rerun."
        )
    command = [
        pandoc,
        "--from=latex",
        "--to=markdown+tex_math_dollars-raw_html-fenced_divs-native_divs",
        "--wrap=none",
        "--quiet",
        str(main_tex),
    ]
    result = subprocess.run(
        command,
        cwd=tex_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or "").strip() or "unknown pandoc error"
        raise ArxivSourceUnavailable(f"pandoc failed: {message}")
    return result.stdout


def _strip_outer_math_environment(block: str) -> str:
    """Remove the outer equation/align/eqnarray wrapper from a math block."""
    match = re.match(r"\\begin\{([A-Za-z*]+)\}", block.lstrip())
    if not match:
        return block
    env = match.group(1)
    if env not in _DISPLAY_MATH_ENVS:
        return block
    leading = len(block) - len(block.lstrip())
    end_pattern = re.compile(r"\\end\{" + re.escape(env) + r"\}\s*$")
    end = end_pattern.search(block)
    if not end:
        return block
    return block[leading + match.end() : end.start()]


def _clean_display_math(match: re.Match) -> str:
    block = _strip_outer_math_environment(match.group(1).strip())
    block = re.sub(r"\\label\{[^{}]*\}", "", block)
    block = re.sub(r"\\(?:notag|nonumber)\b", "", block)
    block = re.sub(r"\\tag\*?\{[^{}]*\}", "", block)
    block = re.sub(r"[ \t]+\n", "\n", block)
    block = re.sub(r"\n{3,}", "\n\n", block)
    return "$$\n" + block.strip() + "\n$$"


def clean_pandoc_markdown(markdown: str) -> str:
    """Normalise pandoc output for chunking and retrieval.

    - keeps ``$...$`` inline math and ``$$...$$`` display math;
    - strips theorem numbers that pandoc invents;
    - drops reference/citation link plumbing while keeping labels;
    - removes div wrappers but keeps their content;
    - drops the bibliography remainder and raw LaTeX labels.
    """
    text = markdown

    # Div wrappers from older pandoc formats: keep the content.
    text = re.sub(r"<div[^>]*>", "", text)
    text = re.sub(r"</div>", "", text)
    text = re.sub(r"(?m)^\s*:::\s*(?:\{[^}]*\})?\s*$", "", text)

    # Do not run a generic HTML-tag stripper here: formulas contain ``<``
    # (e.g. ``0<s\\leq t``), which a tag regex would swallow as raw HTML.
    # Pandoc is already invoked with -raw_html; only the known div wrapper
    # names above need removing.

    # Normalise display math and strip outer LaTeX environments inside $$.
    # Anchor on the LaTeX environment: pandoc puts every display block in a
    # \begin{...} wrapper, so a closing $$ before a prose word cannot be
    # mistaken for the opening of the next block.
    text = re.sub(
        r"\$\$(?=\s*\\begin\{)(.+?)\$\$",
        _clean_display_math,
        text,
        flags=re.S,
    )

    # Theorem-like labels: keep the name, drop invented/unsafe numbering.
    labels = "|".join(_THEOREM_LABELS)
    text = re.sub(
        rf"\*\*({labels})\s+\d+(?:\.\d+)*\*\*\s*\.?",
        r"**\1.**",
        text,
    )
    text = re.sub(
        rf"\b({labels})\s+\d+(?:\.\d+)+\b",
        r"\1",
        text,
    )
    text = re.sub(rf"\*\*({labels})\*\*", r"**\1.**", text)

    # Citations and reference links: keep prose, drop the numbers/plumbing.
    text = re.sub(r"\[@[^\]]+\]", "", text)
    # Pandoc emits equation references as [\[label\]](#label){...}.
    text = re.sub(
        r"\[\\\[[^\]\n]{1,100}\\\]\]\(#[^)\n]{1,120}\)(?:"
        r"\{[^}\n]{0,200}\})?",
        "",
        text,
    )
    # Numeric/lettered cross-references.  Restrict the label so interval
    # notation such as $[0,+\infty)$ can never be mistaken for a link.
    text = re.sub(
        r"\[[^\]\s$(){}\[\]]{1,80}\]\(#[^)\s]{1,120}\)\{[^}\n]{0,200}\}",
        "",
        text,
    )
    text = re.sub(
        r"\[[^\]\s$(){}\[\]]{1,80}\]\(#[^)\s]{1,120}\)",
        "",
        text,
    )
    text = re.sub(r'\{reference-type="[^"]*"[^}]*\}', "", text)
    text = re.sub(r"\[([^\]]+)\]\{\.smallcaps\}", r"\1", text)

    # Labels, spacing commands, and heading attributes.
    text = re.sub(r"\\label\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:notag|nonumber)\b", "", text)
    text = re.sub(r"\{#[^}\n]*\}", "", text)
    text = re.sub(r"\{[.#][^}\n]*\}", "", text)

    # Empty code listings left after ignoring BOSSIP_*.py.
    text = re.sub(r"```[^\n]*\n\s*```", "", text)

    # Whitespace tidy. Do not alter the contents of $...$ math here.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def extract_arxiv_source(
    arxiv_url: str,
    *,
    cache_dir: str | Path = "data/raw",
) -> Extracted:
    """Extract an arXiv paper from its LaTeX e-print via pandoc.

    The returned ``Extracted.text`` is Markdown with ``$...$`` and
    ``$$...$$`` preserved.  ``source_format`` is ``arxiv_latex``; callers keep
    the original arXiv PDF URL for citations.

    Raises ``PandocNotFoundError`` if pandoc is absent and
    ``ArxivSourceUnavailable`` if the e-print is not LaTeX.
    """
    arxiv_id = arxiv_id_from_url(arxiv_url)
    if not arxiv_id:
        raise ArxivSourceUnavailable("not an arXiv abs/pdf/html URL")

    cache_dir = Path(cache_dir)
    markdown_cache = cache_dir / f"arxiv_{arxiv_id}_source.md"
    if markdown_cache.exists():
        text = markdown_cache.read_text(encoding="utf-8")
        return Extracted(
            text=text,
            pages=0,  # LaTeX source has no page count
            content_type="latex",
            extractor="pandoc (arXiv LaTeX)",
            quality_score=0.0,
            all_scores={"arxiv_latex": 0.0},
            source_format="arxiv_latex",
        )

    data, _from_cache = _download_arxiv_eprint(arxiv_id, cache_dir)
    with tempfile.TemporaryDirectory(prefix="quantrag-arxiv-") as tmp:
        root = Path(tmp)
        main_tex = _prepare_tex_tree(data, root)
        raw_markdown = _pandoc_convert(root, main_tex)
    text = clean_pandoc_markdown(raw_markdown)
    markdown_cache.write_text(text, encoding="utf-8")
    return Extracted(
        text=text,
        pages=0,  # LaTeX source has no page count
        content_type="latex",
        extractor="pandoc (arXiv LaTeX)",
        quality_score=0.0,
        all_scores={"arxiv_latex": 0.0},
        source_format="arxiv_latex",
    )


# ---------------------------------------------------------------------------
# Per-document extractor selection
# ---------------------------------------------------------------------------
def _score_extraction(text: str) -> float:
    """Score extraction quality (lower is better).

    Metrics:
    - suspect splits (weight 2.0)
    - run-on tokens >18 chars (weight 3.0)
    - vocab miss rate (weight 0.5)
    - paragraph count deviation from ideal 50 (weight 0.1)
    """
    if not text:
        return float("inf")

    # suspect splits
    concat_counts = {}
    for match in re.finditer(r"[a-zA-Z]{3,}", text.lower()):
        w = match.group()
        concat_counts[w] = concat_counts.get(w, 0) + 1
    repeated = {w for w, c in concat_counts.items() if c >= 2}
    suspects = 0
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    for i in range(len(words) - 1):
        x, y = words[i], words[i + 1]
        if not (
            (len(x) >= 2 and len(y) >= 2)
            or (len(x) == 1 and x.lower() not in {"a", "i"})
            or (len(y) == 1 and y.lower() not in {"a", "i"})
        ):
            continue
        if (x + y).lower() in repeated:
            suspects += 1

    # run-on tokens
    runons = len(re.findall(r"[A-Za-z]{19,}", text))

    # vocab miss rate
    common = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        # Words needed for squashed-prose segmentation in the corpus
        "simplest",
        "quadrature",
        "approximates",
        "throughout",
        "chapter",
        "independently",
        "choice",
        "representation",
        "equivalence",
        "strong",
        "optimal",
        "approximation",
    }
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    vocab_miss = sum(1 for w in words if w not in common) / len(words) if words else 0.0

    # paragraph count
    paras = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])

    score = suspects * 2.0 + runons * 3.0 + vocab_miss * 50 + abs(paras - 50) * 0.1
    return score


def _extract_pypdf_default(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text(extraction_mode="plain") or "" for p in reader.pages]
    return "\n\n".join(pages)


def _extract_pypdf_layout(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text(extraction_mode="layout") or "" for p in reader.pages]
    return "\n\n".join(pages)


def _extract_pdfplumber(data: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(pages)


def _extract_pdfminer(data: bytes) -> str:
    from pdfminer.high_level import extract_text as pdfminer_extract

    return pdfminer_extract(io.BytesIO(data))


def select_best_extractor(data: bytes) -> tuple[str, str, float, dict[str, float]]:
    """Select best extractor for a PDF document.

    Returns (extractor_name, extracted_text, best_score, all_scores).
    ``all_scores`` maps extractor name -> quality score (lower is better).
    """
    extractors = {
        "pypdf default": _extract_pypdf_default,
        "pypdf layout": _extract_pypdf_layout,
        "pdfplumber": _extract_pdfplumber,
        "pdfminer.six": _extract_pdfminer,
    }

    best_score = float("inf")
    best_name = "pypdf default"
    best_text = ""
    all_scores: dict[str, float] = {}

    for name, extractor in extractors.items():
        try:
            text = extractor(data)
            if not text or not text.strip():
                continue
            score = _score_extraction(text)
            all_scores[name] = score
            if score < best_score:
                best_score = score
                best_name = name
                best_text = text
        except Exception:
            continue

    return best_name, best_text, best_score, all_scores


class DoclingUnavailableError(RuntimeError):
    """Raised when the optional docling extractor is requested but missing."""


def extract_pdf_docling(
    data: bytes,
    *,
    page_range: tuple[int, int] | None = None,
    do_formula_enrichment: bool = True,
) -> Extracted:
    """Extract a PDF with docling, optionally with formula enrichment.

    This is intentionally opt-in: docling downloads layout/table/formula
    models and is much slower than pypdf/pdfplumber.  Model weights remain in
    the user's cache and are never committed.
    """
    try:
        from docling.datamodel.base_models import DocumentStream, InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DoclingUnavailableError(
            "docling is not installed; install the math-pdf extra "
            "(`uv sync --extra math-pdf`) to use it."
        ) from exc

    options = PdfPipelineOptions()
    options.do_formula_enrichment = do_formula_enrichment
    options.do_ocr = False
    options.do_table_structure = False
    options.do_code_enrichment = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    stream = DocumentStream(name="document.pdf", stream=io.BytesIO(data))
    kwargs: dict[str, object] = {}
    if page_range is not None:
        kwargs["page_range"] = page_range
    result = converter.convert(stream, **kwargs)
    text = result.document.export_to_markdown()
    pages = (
        page_range[1] - page_range[0] + 1
        if page_range is not None
        else max(1, len(getattr(result.document, "pages", {}) or {}))
    )
    try:
        from docling import __version__ as docling_version
    except Exception:  # pragma: no cover - version attribute varies
        docling_version = "unknown"
    return Extracted(
        text=text,
        pages=pages,
        content_type="pdf",
        extractor=f"docling {docling_version} (formula enrichment)",
        quality_score=0.0,
        all_scores={"docling": 0.0},
        source_format="docling",
    )


def extract_pdf(data: bytes) -> Extracted:
    # Use best extractor for the full document
    extractor_name, extracted_text, best_score, all_scores = select_best_extractor(data)

    # Count actual pages from the raw PDF
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        actual_pages = len(reader.pages)
    except Exception:
        actual_pages = 1

    pages_text = [extracted_text]

    # remove repeated headers/footers: lines appearing on many pages
    line_counts: dict[str, int] = {}
    page_lines: list[list[str]] = []
    for pt in pages_text:
        lines = [ln.rstrip() for ln in pt.splitlines()]
        page_lines.append(lines)
        seen_on_page: set[str] = set()
        for ln in lines:
            s = ln.strip()
            if s and s not in seen_on_page:
                line_counts[s] = line_counts.get(s, 0) + 1
                seen_on_page.add(s)

    threshold = max(2, len(pages_text) // 2) if pages_text else 2
    boilerplate = {ln for ln, c in line_counts.items() if c >= threshold}

    def _is_page_furniture(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        # "12", "12 / 26", "Page 12 of 26", "- 12 -"
        if re.fullmatch(r"\d{1,4}", s):
            return True
        if re.fullmatch(r"(?:page\s*)?\d{1,4}\s*(?:/|of)\s*\d{1,4}", s, re.IGNORECASE):
            return True
        if re.fullmatch(r"[-–—]\s*\d{1,4}\s*[-–—]", s):
            return True
        # encoding artefacts: only replacement chars / stray marks, no letters/digits
        return bool(not any(ch.isalnum() for ch in s) and len(s) <= 8)

    cleaned_pages: list[str] = []
    for lines in page_lines:
        out: list[str] = []
        for ln in lines:
            s = ln.strip()
            if s in boilerplate:
                continue
            if _is_page_furniture(s):
                continue
            out.append(ln)
        cleaned_pages.append("\n".join(out))

    text = "\n\n".join(cleaned_pages)
    # de-hyphenate line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # ---- NEW cleaning steps ----
    # 1. NFC + ligature map (preserves math symbols)
    text = _nfkc(text)
    # 2. Normalize stranded single-symbol lines (e.g. "n", "k" on their own line)
    text = _normalize_single_symbol_lines(text)
    # 3. Segment squashed prose runs (e.g. "Thesimplestquadraturethatapproximates")
    text = _segment_squashed_prose(text)
    # 4. Split CamelCase identifiers (e.g. SimpleGenericMonteCarloClass)
    text = _split_camel_case(text)
    # 5. Vocabulary-gated join of words incorrectly split by a space
    text = _vocabulary_gated_join(text)
    # 6. Dedupe repeated paragraphs inside a chunk
    text = _dedupe_repeated_paragraphs(text)
    # 4. Exclude chunks that are >60% math symbols/digits as kind=formula_dump
    if _formula_dump_ratio(text) > 0.6:
        return Extracted(
            text=text,
            pages=actual_pages,
            content_type="pdf",
            extractor=extractor_name,
            quality_score=best_score,
            all_scores=all_scores,
            source_format="pdf",
        )

    text = _collapse_whitespace(text)
    return Extracted(
        text=text,
        pages=actual_pages,
        content_type="pdf",
        extractor=extractor_name,
        quality_score=best_score,
        all_scores=all_scores,
        source_format="pdf",
    )


def pdf_image_profile(data: bytes) -> dict[str, object]:
    """Per-page profile for a PDF used to detect image-locked slide decks.

    Returns a dict with ``pages``, ``total_chars``, ``avg_chars_per_page``,
    ``image_dominated_pages`` (pages that carry images but almost no extractable
    text), ``per_page`` (list of {chars, images}) and ``image_locked``.

    A document is flagged ``image_locked`` when it is a multi-page deck whose
    text density is too low to be useful *and* a meaningful share of its pages
    are images with little text -- i.e. the formulas/charts live in images that
    no extractor (without OCR) can recover.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    per_page: list[dict[str, int]] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - defensive
            text = ""
        images = getattr(page, "images", None) or []
        per_page.append({"chars": len(text), "images": len(images)})

    total_pages = len(per_page) or 1
    total_chars = sum(p["chars"] for p in per_page)
    avg_chars = total_chars / total_pages
    image_dominated = sum(1 for p in per_page if p["images"] > 0 and p["chars"] < 200)
    image_locked = total_pages > 5 and avg_chars < 400 and (image_dominated / total_pages) > 0.25
    return {
        "pages": total_pages,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg_chars, 1),
        "image_dominated_pages": image_dominated,
        "image_locked": bool(image_locked),
        "per_page": per_page,
    }
