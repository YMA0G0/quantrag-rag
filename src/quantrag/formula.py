"""Formula-aware tokenization and normalization for quantitative retrieval.

This module provides tokenization that preserves mathematical structure by
normalizing unicode math symbols to canonical forms and extracting
mathematical expressions as distinct tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Unicode math symbols mapping to canonical ASCII representations
MATH_SYMBOL_MAP = {
    # Greek letters (lowercase)
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    # Greek letters (uppercase)
    "Α": "Alpha",
    "Β": "Beta",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Ε": "Epsilon",
    "Ζ": "Zeta",
    "Η": "Eta",
    "Θ": "Theta",
    "Ι": "Iota",
    "Κ": "Kappa",
    "Λ": "Lambda",
    "Μ": "Mu",
    "Ν": "Nu",
    "Ξ": "Xi",
    "Ο": "Omicron",
    "Π": "Pi",
    "Ρ": "Rho",
    "Σ": "Sigma",
    "Τ": "Tau",
    "Υ": "Upsilon",
    "Φ": "Phi",
    "Χ": "Chi",
    "Ψ": "Psi",
    "Ω": "Omega",
    # Operators and symbols (excluding dt, dB, dW, dX, dS,
    # dτ, dλ which are handled as differentials)
    "∫": "integral",
    "∮": "contour_integral",
    "∬": "double_integral",
    "∭": "triple_integral",
    "∑": "sum",
    "∏": "product",
    "∐": "coproduct",
    "∂": "partial",
    "∇": "nabla",
    "∆": "delta",
    "√": "sqrt",
    "∛": "cbrt",
    "∜": "fourth_root",
    "∞": "infty",
    "∝": "propto",
    "∼": "sim",
    "≃": "simeq",
    "≈": "approx",
    "≅": "cong",
    "≡": "equiv",
    "≤": "le",
    "≥": "ge",
    "≠": "ne",
    "≪": "ll",
    "≫": "gg",
    "±": "pm",
    "∓": "mp",
    "×": "times",
    "÷": "div",
    "·": "cdot",
    "∘": "circ",
    "•": "bullet",
    "∈": "in",
    "∉": "notin",
    "∋": "ni",
    "∌": "notni",
    "⊂": "subset",
    "⊃": "supset",
    "⊆": "subseteq",
    "⊇": "supseteq",
    "⊄": "nsubset",
    "⊅": "nsupset",
    "∪": "cup",
    "∩": "cap",
    "∅": "emptyset",
    "∃": "exists",
    "∄": "nexists",
    "∀": "forall",
    "∴": "therefore",
    "∵": "because",
    "⇒": "implies",
    "⇐": "impliedby",
    "⇔": "iff",
    "→": "to",
    "←": "leftarrow",
    "↔": "iff",
    "↑": "uparrow",
    "↓": "downarrow",
    "↕": "updown",
    "⇑": "Uparrow",
    "⇓": "Downarrow",
    "⇕": "Updownarrow",
    "∥": "parallel",
    "⊥": "perp",
    "∠": "angle",
    "°": "deg",
    "′": "prime",
    "″": "doubleprime",
    "‴": "tripleprime",
    "…": "cdots",
    "⋯": "cdots",
    "⋮": "vdots",
    "⋱": "ddots",
    "∰": "oiiint",
    "□": "square",
    "△": "triangle",
    "⊕": "oplus",
    "⊗": "otimes",
    "⊙": "odot",
    "⊖": "ominus",
    "⊘": "oslash",
    "⌈": "lceil",
    "⌉": "rceil",
    "⌊": "lfloor",
    "⌋": "rfloor",
    "⟨": "langle",
    "⟩": "rangle",
    "ℕ": "N",
    "ℤ": "Z",
    "ℚ": "Q",
    "ℝ": "R",
    "ℂ": "C",
    "ℍ": "H",
    "ℙ": "P",
    "𝔼": "E",
    "𝕍": "V",
    "ℋ": "H",
    "ℒ": "L",
    "ℐ": "I",
    "ℑ": "Im",
    "ℜ": "Re",
    "ℨ": "Z",
    "≜": "coloneqq",
    "≝": "eqqcolon",
    "↦": "mapsto",
    "↣": "hookrightarrow",
    "↠": "twoheadrightarrow",
}

# Build regex for math symbols (longer symbols first to avoid partial matches)
MATH_SYMBOLS_SORTED = sorted(MATH_SYMBOL_MAP.keys(), key=len, reverse=True)
MATH_SYMBOL_PATTERN = re.compile("|".join(re.escape(s) for s in MATH_SYMBOLS_SORTED))

# LaTeX-like patterns
LATEX_PATTERNS = [
    (re.compile(r"\\(?:frac|sqrt|partial|int|sum|prod|lim|inf|sup)\b", re.ASCII), "latex_cmd"),
    (re.compile(r"\\[a-zA-Z]+\b", re.ASCII), "latex_sym"),
    (re.compile(r"\^[\d\w]+", re.ASCII), "superscript"),
    (re.compile(r"_[\d\w]+", re.ASCII), "subscript"),
    (re.compile(r"\\[{}\[\]]", re.ASCII), "latex_brace"),
]

# Differential pattern: d followed by single letter (dt, dB, dW, etc.)
DIFFERENTIAL_PATTERN = re.compile(r"\b[dδ][A-Za-zτλ]\b", re.ASCII)

# Variable-like patterns: single letters with optional subscripts
VARIABLE_PATTERN = re.compile(r"[A-Za-z](?:_[A-Za-z0-9]+)?(?![A-Za-z0-9_])", re.ASCII)

# Numbers with units or math context
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b", re.ASCII)

# Word pattern (ASCII only)
WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)


@dataclass(frozen=True, slots=True)
class FormulaToken:
    """A token with its type for formula-aware processing."""

    text: str
    token_type: str  # "word", "math_symbol", "latex", "differential", "variable", "number"
    original: str


def normalize_math_symbols(text: str) -> str:
    """Replace unicode math symbols with canonical ASCII representations."""

    def replace_match(match):
        return MATH_SYMBOL_MAP.get(match.group(0), match.group(0))

    return MATH_SYMBOL_PATTERN.sub(replace_match, text)


def normalize_query_for_retrieval(query: str) -> str:
    """
    Normalize a query string for retrieval by expanding math symbols
    and normalizing whitespace.
    """
    normalized = normalize_math_symbols(query)
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _token_scanner(text: str):
    r"""Greedy left-to-right scanner returning (text, token_type, original) per token.

    Alternatives are tried in priority order at each position:
      1. math symbols (longest first)   -> math_symbol
      2. differentials (d/delta + letter) -> differential
      3. LaTeX commands (\cmd)          -> latex_cmd
      4. variables with subscript (X_t) -> variable
      5. superscripts (^2)              -> superscript
      6. subscripts (_1)                -> subscript
      7. numbers                        -> number
      8. ASCII words                    -> word
      9. any remaining non-space char   -> punct
    """
    alternation = "|".join(
        [
            r"|".join(re.escape(s) for s in MATH_SYMBOLS_SORTED),
            r"[dδ][A-Za-zτλ](?![A-Za-z0-9_])",
            r"\\[A-Za-z]+",
            r"[A-Za-z]_[A-Za-z0-9]+",
            r"\^[A-Za-z0-9]+",
            r"_[A-Za-z0-9]+",
            r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
            r"[A-Za-z][A-Za-z0-9]*",
            r"[^\s]",
        ]
    )
    pattern = re.compile(alternation)
    for match in pattern.finditer(text):
        original = match.group(0)
        if original in MATH_SYMBOL_MAP:
            yield MATH_SYMBOL_MAP[original].lower(), "math_symbol", original
            continue
        if re.fullmatch(r"[dδ][A-Za-zτλ](?![A-Za-z0-9_])", original):
            yield original, "differential", original
            continue
        if original.startswith("\\"):
            yield original, "latex_cmd", original
            continue
        if original.startswith("^"):
            yield original, "superscript", original
            continue
        if original.startswith("_"):
            yield original, "subscript", original
            continue
        if re.fullmatch(r"[A-Za-z]_[A-Za-z0-9]+", original):
            yield original, "variable", original
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", original):
            yield original, "number", original
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", original):
            yield original.lower(), "word", original
            continue
        yield original, "punct", original


def extract_math_tokens(text: str) -> list[FormulaToken]:
    """Tokenize ``text`` into typed math-aware tokens.

    Each token carries its ``text`` (normalized), ``token_type`` (one of
    ``word``, ``math_symbol``, ``latex``, ``differential``, ``variable``,
    ``number``, ``superscript``, ``subscript``), and the ``original``
    substring from the source.
    """
    tokens: list[FormulaToken] = []
    for ttext, ttype, original in _token_scanner(text):
        tokens.append(FormulaToken(text=ttext, token_type=ttype, original=original))
    return tokens


def tokenize_for_retrieval(text: str) -> list[str]:
    """Return normalized token strings for retrieval indexing.

    Math symbols become canonical ASCII names; ASCII words are lowercased;
    differentials keep their original case; numbers and variables are kept
    as-is. Pure punctuation is dropped.
    """
    out: list[str] = []
    for ttext, ttype, _original in _token_scanner(text):
        if ttype == "punct":
            continue
        out.append(ttext)
    return out


def fast_tokenize(text: str) -> list[str]:
    """Fast tokenization for BM25 indexing.

    Applies math-symbol normalization then splits on word boundaries.
    Returns tokens suitable for term-frequency counting; differentials keep
    case, ASCII words are lowercased, pure punctuation is dropped.
    """
    return tokenize_for_retrieval(text)


def compute_term_frequencies(tokens: list[str]) -> dict[str, int]:
    """Return ``{token: count}`` for the given token list."""
    tf: dict[str, int] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return tf


def extract_formula_expressions(text: str) -> list[str]:
    """Extract math expression sequences from ``text``.

    A formula expression is a run of consecutive formula-ish segments
    (math symbols, differentials, operators, numbers, uppercase variables)
    joined across whitespace. Runs of a single bare token with no operator
    (e.g. a lone "X") are not considered formulas, so plain English text
    yields no expressions.
    """
    normalized = normalize_math_symbols(text)
    segments = normalized.split()
    operators = set("=+-*/^()[]{}<>,;:.")
    math_terms = {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "theta",
        "lambda",
        "mu",
        "sigma",
        "rho",
        "nu",
        "pi",
        "omega",
        "phi",
        "partial",
        "integral",
        "sum",
        "product",
        "nabla",
        "infty",
    }

    def is_mathish(seg: str) -> bool:
        if any(ch in operators for ch in seg):
            return True
        if any(ch.isdigit() for ch in seg):
            return True
        if seg != seg.lower() or any(ch.isupper() for ch in seg):
            return True
        if seg in math_terms:
            return True
        return bool(re.fullmatch(r"[dδ][A-Za-zτλ]", seg))

    expressions: list[str] = []
    current: list[str] = []
    for seg in segments:
        if is_mathish(seg):
            current.append(seg)
        else:
            if len(current) >= 2 or any(any(ch in operators for ch in s) for s in current):
                expressions.append(" ".join(current))
            current = []
    if len(current) >= 2 or any(any(ch in operators for ch in s) for s in current):
        expressions.append(" ".join(current))
    return expressions
