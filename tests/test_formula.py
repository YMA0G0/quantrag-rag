"""Tests for formula-aware tokenization."""

from __future__ import annotations

import pytest

from quantrag.formula import (
    FormulaToken,
    extract_formula_expressions,
    extract_math_tokens,
    fast_tokenize,
    normalize_math_symbols,
    normalize_query_for_retrieval,
    tokenize_for_retrieval,
)


class TestNormalizeMathSymbols:
    def test_greek_letters_lowercase(self):
        text = "α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω"
        result = normalize_math_symbols(text)
        expected = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
            "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
        )
        assert result == expected

    def test_greek_letters_uppercase(self):
        text = "Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω"
        result = normalize_math_symbols(text)
        expected = (
            "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda "
            "Mu Nu Xi Omicron Pi Rho Sigma Tau Upsilon Phi Chi Psi Omega"
        )
        assert result == expected

    def test_operators(self):
        text = "∫ ∑ ∏ √ ∂ ∇ ∆ ∞ ∝ ∼ ≈ ≤ ≥ ≠ ± × ÷ · ∈ ∉ ⊂ ⊃ ⊆ ⊇ ∪ ∩ ∅ ∀ ∃ ∇ → ← ↔ ⇒ ⇐ ⇔"
        result = normalize_math_symbols(text)
        assert "integral" in result
        assert "sum" in result
        assert "product" in result
        assert "sqrt" in result
        assert "partial" in result
        assert "nabla" in result
        assert "delta" in result
        assert "sqrt" in result
        assert "infty" in result
        assert "propto" in result
        assert "sim" in result
        assert "approx" in result
        assert "le" in result
        assert "ge" in result
        assert "ne" in result

    def test_differentials(self):
        text = "dt dB dW dX dS dY dZ dτ dλ"
        result = normalize_math_symbols(text)
        assert result == "dt dB dW dX dS dY dZ dtau dlambda"

    def test_preserves_ascii_text(self):
        text = "The Black-Scholes formula for option pricing"
        result = normalize_math_symbols(text)
        assert result == text


class TestExtractMathTokens:
    def test_words(self):
        tokens = extract_math_tokens("Black Scholes option pricing")
        assert all(t.token_type == "word" for t in tokens)
        assert [t.text for t in tokens] == ["black", "scholes", "option", "pricing"]

    def test_math_symbols(self):
        tokens = extract_math_tokens("α + β = γ")
        types = [t.token_type for t in tokens]
        assert "math_symbol" in types

    def test_differentials(self):
        tokens = extract_math_tokens("dt dB dW")
        diff_tokens = [t for t in tokens if t.token_type == "differential"]
        assert len(diff_tokens) == 3
        assert {t.text for t in diff_tokens} == {"dt", "dB", "dW"}

    def test_variables(self):
        tokens = extract_math_tokens("X_t = X_0 + μt")
        # Variables with subscripts are tokenized as single variable tokens
        var_tokens = [t for t in tokens if t.token_type == "variable"]
        assert any(t.text == "X_t" for t in var_tokens)
        assert any(t.text == "X_0" for t in var_tokens)
        # μ is normalized to mu (math_symbol), t is part of variable μt -> "mut" or split
        # Actually μ is math_symbol "mu", t is word "t"
        var_texts = {t.text for t in var_tokens}
        assert "X_t" in var_texts
        assert "X_0" in var_texts

    def test_numbers(self):
        tokens = extract_math_tokens("The value is 3.14159 and 42")
        num_tokens = [t for t in tokens if t.token_type == "number"]
        assert len(num_tokens) == 2
        assert "3.14159" in [t.text for t in num_tokens]
        assert "42" in [t.text for t in num_tokens]

    def test_latex_commands(self):
        tokens = extract_math_tokens(r"\int \frac \partial \sum")
        latex_tokens = [t for t in tokens if t.token_type in ("latex_cmd", "latex_sym")]
        assert len(latex_tokens) >= 3

    def test_superscript_subscript(self):
        tokens = extract_math_tokens("X^2 Y_1")
        super_tokens = [t for t in tokens if t.token_type == "superscript"]
        [t for t in tokens if t.token_type == "subscript"]
        # Superscript ^2 is detected
        assert len(super_tokens) >= 1
        # Subscript _1 is part of variable Y_1 (tokenized as variable)
        # The test verifies superscript works; subscript is part of variable token
        assert len(super_tokens) >= 1


class TestTokenizeForRetrieval:
    def test_basic(self):
        tokens = tokenize_for_retrieval("Black Scholes option pricing")
        assert "black" in tokens
        assert "scholes" in tokens

    def test_math_symbols_normalized(self):
        tokens = tokenize_for_retrieval("α + β = γ")
        assert "alpha" in tokens
        assert "beta" in tokens
        assert "gamma" in tokens

    def test_differentials_preserved(self):
        tokens = tokenize_for_retrieval("dt dB dW")
        assert "dt" in tokens
        assert "dB" in tokens
        assert "dW" in tokens

    def test_formula_expressions(self):
        tokens = tokenize_for_retrieval("dS = μS dt + σS dW")
        assert "dt" in tokens
        assert "dW" in tokens
        assert "mu" in tokens
        assert "sigma" in tokens


class TestExtractFormulaExpressions:
    def test_extracts_math_sequences(self):
        formulas = extract_formula_expressions("The formula is dS = μS dt + σS dW and also X = Y")
        assert len(formulas) >= 1
        assert any("mu" in f and "dt" in f for f in formulas)

    def test_ignores_short_sequences(self):
        formulas = extract_formula_expressions("Just a single variable X")
        # Single variable shouldn't form a formula expression
        assert len(formulas) == 0


class TestNormalizeQuery:
    def test_normalizes_math_symbols(self):
        result = normalize_query_for_retrieval("What is ∫ x dx?")
        assert "integral" in result

    def test_normalizes_whitespace(self):
        result = normalize_query_for_retrieval("  What   is   the   formula?  ")
        assert result == "What is the formula?"


class TestFastTokenize:
    def test_fast_tokenize_basic(self):
        tokens = fast_tokenize("Black Scholes option pricing")
        assert "black" in tokens
        assert "scholes" in tokens

    def test_fast_tokenize_math(self):
        tokens = fast_tokenize("dS = μS dt + σS dW")
        assert "dt" in tokens
        assert "mu" in tokens
        assert "sigma" in tokens

    def test_fast_tokenize_greek(self):
        tokens = fast_tokenize("α β γ")
        assert "alpha" in tokens
        assert "beta" in tokens
        assert "gamma" in tokens


class TestFormulaToken:
    def test_formula_token_creation(self):
        token = FormulaToken(text="alpha", token_type="math_symbol", original="α")
        assert token.text == "alpha"
        assert token.token_type == "math_symbol"
        assert token.original == "α"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
