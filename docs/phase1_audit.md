# Phase 1 / 1.5 — detailed audit report

Generated read-only from the final artifacts after the deterministic pilot rerun.

- Pilot command: `uv run python -m quantrag.ingest --profile core --max-tier 1 --limit 10`
- Pilot rerun (UTC): `2026-10-23T19:42:42Z`
- Artifacts written (local +03:00): `2026-10-23 22:42:49`
- Tokenizer: `quantrag.ingest.chunking.count_tokens` → `tiktoken` `cl100k_base`; hard cap `MAX_TOKENS_HARD = 900`.

## 0. Scope reconciliation: manifest vs profile selection vs pilot

| Scope | Rows | Composition |
|---|---|---|
| Full manifest (2 CSVs) | 120 | 119 unique URLs + 1 duplicate URL row |
| `--profile core` selection | 120 | 24 selected, 96 excluded (`outside_core_profile` 39, `unsupported_resource_type` 17, `landing_or_marketing` 14, `not_extractable` 11, `unsupported_content_type` 7, `needs_manual_review` 3, `needs_review` 3, `duplicate_url` 1, `not_found` 1) |
| `+ --max-tier 1` | 120 | 18 eligible, 6 of the 24 core selections become `tier_filtered` |
| `+ --limit 10` (pilot) | 120 | **10 processed**, 8 deferred (`limit`) |
| Pilot outcomes | 10 | **9 ingested, 1 failed, 0 skipped** |
| Report rows | 120 | 9 ingested + 1 failed + 110 skipped |

The 110 skipped report rows are *not* pilot outcomes: they are 96 core-profile exclusions + 6 tier-filtered + 8 limit-deferred. Therefore `9 + 110 + 1 = 120` is a **manifest/report-row** identity; the pilot identity is `9 ingested + 1 failed + 0 skipped = 10`.

## 1. The ten pilot rows

| # | status | reason | chunks | pages | characters | original (citation) URL | used URL |
|---|---|---|---|---|---|---|---|
| 1 | ingested | ok | 55 | 84 | 45072 | https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/resources/mit18_642_f24_lec19_1_pdf/ | https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/mit18_642_f24_lec19_1.pdf |
| 2 | ingested | ok | 28 | 75 | 37830 | https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/resources/mit15_450f10_lec02/ | https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/0d1260b891a96241316d883d4f5bfaec_MIT15_450F10_lec02.pdf |
| 3 | ingested | ok | 7 | 8 | 11027 | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote5/ | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/f5784e4facf3de690210d17c97358eba_MIT18_S096F13_lecnote5.pdf |
| 4 | ingested | ok | 6 | 7 | 10130 | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote17/ | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/3b97c6b0c282dd9dc024c4c7ffe3fba8_MIT18_S096F13_lecnote17.pdf |
| 5 | ingested | ok | 6 | 7 | 9982 | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote21/ | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/a671d0d2abe626c371fd5850ad670397_MIT18_S096F13_lecnote21.pdf |
| 6 | ingested | ok | 6 | 16 | 8707 | https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/resources/mit15_450f10_review01/ | https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/f62ab115e23aeb722a6cd813dea32f46_MIT15_450F10_review01.pdf |
| 7 | failed | robots_disallowed | — | — | — | https://www.columbia.edu/~ks20/6712-14/6712-14-Notes-B-Scholes.pdf | — |
| 8 | ingested | ok | 1 | 1 | 1696 | https://csustan.csustan.edu/~tom/Lecture-Notes/HighFinance/ | https://csustan.csustan.edu/~tom/Lecture-Notes/HighFinance/ |
| 9 | ingested | ok | 35 | 29 | 54108 | https://arxiv.org/pdf/2502.07766 | https://arxiv.org/pdf/2502.07766 |
| 10 | ingested | ok | 20 | 21 | 23235 | https://bauer.uh.edu/spirrong/interest_rate_models.pdf | https://bauer.uh.edu/spirrong/interest_rate_models.pdf |
| **Σ** | 9 ingested / 1 failed | | **164** | **248** | **201787** | | |

Aggregates recomputed from these rows: chunks **164**, pages **248**, characters **201,787**; ingested 9, failed 1, skipped 0.

## 2. Token statistics and hard-cap proof

| metric | value |
|---|---|
| total chunks | 164 |
| mean | 494.81 |
| median | 482.0 |
| minimum | 98 |
| maximum | 878 |
| > 800 | 10 |
| > 900 | 0 |
| > 1000 | 0 |
| == 900 | 0 |
| violations of `MAX_TOKENS_HARD=900` | 0 |

Every one of the 164 stored `text` values (context/header line included) was measured with the same function the hard-cap validator uses. Maximum = **878 ≤ 900**; ids at max: `7817a9939d-0019`. The report totals object matches this recomputation exactly (total 164, mean 494.81, median 482.0, max 878, >800 10, >1000 0).

## 3. Seven original pilot failures — diagnostic table

| # | URL | HTTP | Content-Type | cached bytes | first 16 bytes (hex / readable) | error class | root cause | final resolution |
|---|---|---|---|---|---|---|---|---|
| 1 | https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/resources/mit18_642_f24_lec19_1_pdf/ | 200 | `text/html` | `3460fe14fd3d9f0f229d15b4787f8cb6f3a3ec6c.bin` (90,977 B) | `3c21646f63747970652068746d6c3e0a` / `<!doctype html>\n` | `PdfStreamError: Stream has ended unexpectedly` | OCW resource page is an HTML wrapper for a PDF-labelled source; old code trusted the manifest hint and fed HTML to pypdf | magic-byte/HTML detection + one-hop PDF link follow; citation URL preserved; now ingested |
| 2 | https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/resources/mit15_450f10_lec02/ | 200 | `text/html` | `5170453106a893d10b4be2b881f082521ae85711.bin` (66,357 B) | `3c21646f63747970652068746d6c3e0a` / `<!doctype html>\n` | `PdfStreamError: Stream has ended unexpectedly` | OCW resource page is an HTML wrapper for a PDF-labelled source; old code trusted the manifest hint and fed HTML to pypdf | magic-byte/HTML detection + one-hop PDF link follow; citation URL preserved; now ingested |
| 3 | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote5/ | 200 | `text/html` | `30ad0a25d95349e687ea1447263ae8a9fb370465.bin` (70,596 B) | `3c21646f63747970652068746d6c3e0a` / `<!doctype html>\n` | `PdfStreamError: Stream has ended unexpectedly` | OCW resource page is an HTML wrapper for a PDF-labelled source; old code trusted the manifest hint and fed HTML to pypdf | magic-byte/HTML detection + one-hop PDF link follow; citation URL preserved; now ingested |
| 4 | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote17/ | 200 | `text/html` | `94944dd7cb5b8f8e882582ad3ef9016f9822c5d6.bin` (70,613 B) | `3c21646f63747970652068746d6c3e0a` / `<!doctype html>\n` | `PdfStreamError: Stream has ended unexpectedly` | OCW resource page is an HTML wrapper for a PDF-labelled source; old code trusted the manifest hint and fed HTML to pypdf | magic-byte/HTML detection + one-hop PDF link follow; citation URL preserved; now ingested |
| 5 | https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote21/ | 200 | `text/html` | `1c1ddd7b56b8324a08ada13e57c082b2d570646d.bin` (70,663 B) | `3c21646f63747970652068746d6c3e0a` / `<!doctype html>\n` | `PdfStreamError: Stream has ended unexpectedly` | OCW resource page is an HTML wrapper for a PDF-labelled source; old code trusted the manifest hint and fed HTML to pypdf | magic-byte/HTML detection + one-hop PDF link follow; citation URL preserved; now ingested |
| 6 | https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/resources/mit15_450f10_review01/ | 200 | `text/html` | `09211fe1002faaecb2c9b9e2443b67aba53b0265.bin` (66,428 B) | `3c21646f63747970652068746d6c3e0a` / `<!doctype html>\n` | `PdfStreamError: Stream has ended unexpectedly` | OCW resource page is an HTML wrapper for a PDF-labelled source; old code trusted the manifest hint and fed HTML to pypdf | magic-byte/HTML detection + one-hop PDF link follow; citation URL preserved; now ingested |
| 7 | https://www.columbia.edu/~ks20/6712-14/6712-14-Notes-B-Scholes.pdf | unavailable | unavailable | unavailable (no cache file) | unavailable (no request made) | `robots_disallowed` | `robots.txt` disallows the path; compliant crawler never fetched it | intentionally not fetched; manual-download path `data/manual_sources.csv`; remains the 1 expected pilot failure |

Unavailable metadata for row 7 is marked as such: HTTP status/Content-Type/first bytes could only be obtained by violating robots.txt. Rows 1–6 headers were re-requested live (status 200, `text/html`) and the first bytes matched the cached `.bin` files.

## 4. Verbatim chunk appendix

### 4.1 Five chunks from ≥3 documents (seed 2024)

#### Chunk 1 — id `7817a9939d-0030` — 837 tokens — `Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)`
Citation URL: https://arxiv.org/pdf/2502.07766

```text
Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)

2*alp*the - the^2 - 4*the*E*(alp - the) + E^2*(alp^2 + 2*alp*the - 3*the^2))*(4*T_ex*alp^4*lam -
16*T_ex*alp^3*lam*the + 24*T_ex*alp^2*lam*the^2 - 16*T_ex*alp*lam*the^3 + 16*T_ex*lam*the^4*E^4*(5*rho^2 +
1) + 4*T_ex*lam*the^4 - 12*alp^4*rho^2 + 3*alp^4 + 48*alp^3*rho^2*the - 20*alp^3*the - 72*alp^2*rho^2*the^2
+ 40*alp^2*the^2 + 48*alp*rho^2*the^3 - 32*alp*the^3 - 12*rho^2*the^4 + 9*the^4 -
8*the*E^3*(8*T_ex^2*alp*lam^2*rho^2*the^2 - 8*T_ex^2*lam^2*rho^2*the^3 + 8*T_ex*alp^2*lam*rho^2*the +
16*T_ex*alp*lam*rho^2*the^2 + 10*T_ex*alp*lam*the^2 - 32*T_ex*lam*rho^2*the^3 - 10*T_ex*lam*the^3 +
4*alp^3*rho^2 + alp^3 + 8*alp^2*rho^2*the + 5*alp^2*the + 4*alp*rho^2*the^2 - 10*alp*the^2 -
36*rho^2*the^3) + E^4*(4*alp^4*rho^2 + alp^4 + 16*alp^3*rho^2*the + 4*alp^3*the + 40*alp^2*rho^2*the^2 +
8*alp^2*the^2 + 80*alp*rho^2*the^3 + 16*alp*the^3 - 292*rho^2*the^4 - 53*the^4) + 4*E^2*( -
16*T_ex^2*alp^2*lam^2*rho^2*the^2 + 32*T_ex^2*alp*lam^2*rho^2*the^3 - 16*T_ex^2*lam^2*rho^2*the^4 -
16*T_ex*alp^3*lam*rho^2*the + 40*T_ex*alp^2*lam*rho^2*the^2 - 6*T_ex*alp^2*lam*the^2 -
16*T_ex*alp*lam*rho^2*the^3 + 12*T_ex*alp*lam*the^3 - 8*T_ex*lam*rho^2*the^4 - 6*T_ex*lam*the^4 -
6*alp^4*rho^2 - alp^4 + 16*alp^3*rho^2*the - 4*alp^3*the + 8*alp^2*rho^2*the^2 + 36*alp^2*the^2 -
```

#### Chunk 2 — id `3d4a999736-0005` — 348 tokens — `Interest Rate Models (Univ. of Houston lecture notes)`
Citation URL: https://bauer.uh.edu/spirrong/interest_rate_models.pdf

```text
Interest Rate Models (Univ. of Houston lecture notes)

3 The Hull-White Model
As noted earlier, traders don’t like the fact that the Vasice k model (or the
CIR model, for that matter) can’t ﬁt the prices of every zero e xactly. So
what to do? What to do?
No problem!: Add an inﬁnite number of degrees of freedom to ﬁt an
inﬁnite number of bond prices!
That’s what Hull-White do. They posit:
drt = (Θ t − art)dt + σdWt
where Θ t is an arbitrary function. As we will see, we can choose Θ t to ﬁt all
bond prices exactly.
As it turns out, working directly with this equation is somew hat cumbersome, so we will transform variables to make the problem isom orphic to one
that is easy to solve–an Orenstein-Uhlenbeck (“OU”) proces s:
dXt = −µdt + σdWt
To solve this SDE, posit:
Xt = a(t)[X0 +
∫ t
b(s)dWs]
Then, applying Ito’s Lemma:
dXt = a′(t)[X0 +
∫ t
b(s)dWs]dt + a(t)b(t)dWt
dXt = a′(t)Xt
a(t) dt + a(t)b(t)dWt
Now we match coeﬃcients. First, we get an ordinary diﬀerenti al equation:
−µ = a′(t)a(t)
```

#### Chunk 3 — id `fddb59e6ac-0000` — 410 tokens — `High Finance - Lecture Notes on Black-Scholes (CSU Stanislaus)`
Citation URL: https://csustan.csustan.edu/~tom/Lecture-Notes/HighFinance/

```text
High Finance - Lecture Notes on Black-Scholes (CSU Stanislaus)

High Finance - Lecture Notes
Fall, 2008
These are some "lecture notes" and related material for a talk I gave recently.
The announcement for the talk looked like this:
Speaker: Dr. Tom Carter, Department of Computer Science, California State University, Stanislaus

Date: Tuesday, November 18, 2008
Time: 4:00 - 5:00 p.m.
Room: P104
Title: Fatal Shore -- Black-Scholes and High Finance
Abstract: This talk will include a derivation/analysis of the Black-Scholes
options pricing model. I will also discuss some concerns about the model,
potential consequences of the use of such models, and some of the relevance of
such models in the current "financial crisis." We seem to have run aground on
some "dark shoals" near the shores of high finance . . .
Econ102.pdf Deriving and solving the Black-Scholes "options pricing model"
 (a non-linear PDE). Also contained here is some background discussion on random
 variables and random walks.
Econ102-discussion.pdf Pdf version of some "power point" notes containing some
 discussion of the assumptions behind the Black-Scholes model, and some related topics . . .
TaxiCab.pdf How is the "discrete world" related to the "continuous world"? In
 particular, looking at "taxicab geometry" . . .
Logistic-continuous.pdf More of -- how is the "discrete world" related to
 the "continuous world"? This time, the discrete vs. the continuous "logistic map" . . .
econ101.pdf Some lecture notes from a prior talk, concerning maximum entropy principles,
 wealth distribution, and power laws . . .
social-insurance.html Some notes excerpted from a discussion about regulation, taxes
 (in particular, dividend taxes), financial institutions, and bailouts past . . .
```

#### Chunk 4 — id `3460fe14fd-0051` — 614 tokens — `18.642 Lecture 19_1: Stochastic Calculus (MIT OCW, Fall 2024)`
Citation URL: https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/resources/mit18_642_f24_lec19_1_pdf/

```text
18.642 Lecture 19_1: Stochastic Calculus (MIT OCW, Fall 2024)

Logreturnon(t , t+∆t]: ln( Pt+∆tPt )∼ N([µ− σ22 ](∆t), σ2∆t)).
Ito’s Formula: Markov Case
Example 6. Geometric Brownian Motion: {Pt , 0 ≤ t < ∞}
stochastic process of an asset price which is an Ito Process
dPt = µPt dt + σPt dBt
where µ and σ > 0 are constant. Pt is generalized Wiener process,
Xt = Pt with parameters: µ(xt , t)= µxt , σ(xt , t)= σxt
Apply Ito’s Lemma to obtain model for G (Pt , t) = ln(Pt )
∂G 1 ∂G 1 ∂2G 1(−1)• = , =0, = .∂Pt Pt ∂t 2 ∂P2 2 P2 t t
• Ito’s Lemma gives 
1 1(−1) 1d ln(Pt ) = µPt + P2 σ2Pt 2 dt + σPt dBtPt 2 Pt  t
= µ − σ2 dt + σdBt2
Log price follows generalized Wiener process:
drift rate =(µ − σ2/2) and variance rate = σ2

Ito’s Formula: Markov Case
Example 6. Geometric Brownian Motion: {Pt , 0 ≤ t < ∞}
stochastic process of an asset price which is an Ito Process
dPt = µPt dt + σPt dBt
where µ and σ > 0 are constant. Pt is generalized Wiener process,
Xt = Pt with parameters: µ(xt , t)= µxt , σ(xt , t)= σxt
Apply Ito’s Lemma to obtain model for G (Pt , t) = ln(Pt )
∂G 1 ∂G 1 ∂2G 1(−1)• = , =0, = .∂Pt Pt ∂t 2 ∂P2 2 P2 t t
• Ito’s Lemma gives 
1 1(−1) 1d ln(Pt ) = µPt + σ2P2 dt + σPt dBtPt 2 P2 t Pt  t
= µ − σ 22 dt + σdBt
Log price follows generalized Wiener process:
drift rate =(µ − σ2/2) and variance rate = σ2
Pt+∆tLog return on (t, t +∆t]: ln( ) ∼ N([µ − σ2 ](∆t), σ2∆t)).Pt 2
```

#### Chunk 5 — id `09211fe100-0002` — 392 tokens — `15.450 Analytics of Finance, Review 1: Arbitrage-free pricing (MIT OCW)`
Citation URL: https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/resources/mit15_450f10_review01/

```text
15.450 Analytics of Finance, Review 1: Arbitrage-free pricing (MIT OCW)

Black-Scholes Model
Arbitrage-free pricing of options by replication.
uropean option with payoff H(ST ).
eplicating portfolio delta is
∂f (t, St )θt = ∂St
∂f (t, S)−r f (t, S) + ∂f (t, S)+ rS∂t 1+∂S 2∂2σ S2 f (t, S)2 = 0∂S2
ith the boundary condition f (T , S) = H(S).
erive the B.-S. PDE using replication arguments:
df (t, St ) =θt dSt + (f (t, St ) −θt St )r dt
E
R
w
D
c© Leonid Kogan ( MIT, Sloan ) Review: Part I 15.450, Fall 2010 6 / 15

Problem
Your colleagues have developed a term structure model that they intend to
se for pricing of interest-rate sensitive securities. Their model is of the
llowing form: they ﬁt the shape of the term structure using a parsimonious
losed-form description, and then describe the evolution of necessary
arameters. For example, one speciﬁcation of the bond yields yτ is
τ 1yt = a + xb t+τ
dxt = −θ(x t − x) dt +σdZt
ou suspect that this model implies arbitrage opportunities. How can you
onvince your colleagues that this is the case?
u
fo
c
p
Y
c
c© Leonid Kogan ( MIT, Sloan ) Review: Part I 15.450, Fall 2010 7 / 15
```

### 4.2 Two longest chunks

#### Longest #1 — id `7817a9939d-0019` — 878 tokens — `Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)`
Citation URL: https://arxiv.org/pdf/2502.07766

```text
Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)

To deal with this problem, we used further approximations of all integrand parts involving such
terms, prior to the analytical integration. In particular, we retorted to the method of binomial
expansion, which is generally specified as follows:
(1 + 𝑦 )𝑔 ≈ ∑ (𝑔𝑛) 𝑦 𝑔
𝑛=0
= 1 + 𝑔𝑦 + 𝑔(𝑔 − 1)2! 𝑦2 + 𝑔(𝑔 − 1)(𝑔 − 2)3! 𝑦3 … (25)
(25) is valid for all nonnegative (not necessarily integer) values of 𝑔 and for |𝑦| < 1.
One possible approach of approximating (𝛼(𝑥))
𝑔
 is then given by :
(𝛼(𝑥))
𝑔
= (𝜃 + (𝛼 − 𝜃) 𝑒−𝜆𝑥)𝑔 = 𝜃𝑔 (1 + (𝛼 − 𝜃)𝜃 𝑒−𝜆𝑥)𝑔 = 𝜃𝑔(1 + 𝑦)𝑔
with 𝑦 ≡
(𝛼−𝜃)
𝜃 𝑒−𝜆𝑥,
leading to:
(𝛼(𝑥))
𝑔
≈ 𝜃𝑔 (1 + 𝑔𝜃−1(𝛼 − 𝜃)𝑒−𝜆𝑥 + 12 𝑔 (𝑔 − 1)𝜃−2(𝛼 − 𝜃)2 𝑒−2𝜆𝑥
+ 16 𝑔 (𝑔 − 1)(𝑔 − 2)𝜃−3(𝛼 − 𝜃)3 𝑒−3𝜆𝑥 … )
Note that , unlike (𝛼(𝑥))
𝑔
, the approximation is linear in exponential terms
𝑒−𝜆𝑥, 𝑒−2𝜆𝑥, 𝑒−3𝜆𝑥, …, which can be integrated easily with respect to 𝑥.
Unfortunately, the validity condition |𝑦| < 1 will be always satisfied only if 𝛼 < 2𝜃, and this
latter condition will be violated quite often in practice, resulting in erroneous approximations,
in particular for short expiries and high instantaneous volatility8. To make the expansion a
workable solution for all plausible parameter values, we need to reparametrize 𝛼(𝑥) as follows:
𝛼(𝑥) = 𝜃 + (𝛼 − 𝜃) 𝑒−𝜆𝑥 = 𝜇 + 𝜇𝛿 ( 2𝑒−𝜆𝑥 − 1)
with:
𝜇 = 𝛼 + 𝜃2 and 𝛿 = 𝛼 − 𝜃𝛼 + 𝜃
8 In particular, if 𝛼 > 2𝜃 the condition |𝑦| < 1 will be violated for all time points 𝑥 < 1𝜆 𝑙𝑛 (𝛼−𝜃)𝜃 . Another simple
binomial expansion for (𝛼(𝑥))
𝑔
, with factoring out 𝛼 instead of 𝜃, leads to the analogous validity condition 𝜃 <
2𝛼, which will also be violated above a certain 𝑥 in situations of low instantaneous volatility.
```

#### Longest #2 — id `7817a9939d-0033` — 864 tokens — `Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)`
Citation URL: https://arxiv.org/pdf/2502.07766

```text
Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)

mu^3*nu^2*(30*T_ex*lam*E^6*(dlt^6*rho^2 + 12*dlt^5*rho^2 - 12*dlt^4*rho^2 - 1024*dlt^3*rho^2 - 256*dlt^3 +
3072*dlt^2*rho^2 + 768*dlt^2 - 3072*dlt*rho^2 - 768*dlt + 1024*rho^2 + 256) - 80*dlt^6*rho^2 +
192*dlt^4*rho^2*E*(3*dlt^2 + 6*dlt - 8) - 30*dlt^2*E^2*( - 12*T_ex*dlt^4*lam*rho^2 -
48*T_ex*dlt^3*lam*rho^2 + 384*T_ex*dlt^2*lam*rho^2 + 47*dlt^4*rho^2 + 168*dlt^3*rho^2 + 544*dlt^2*rho^2 -
96*dlt*rho^2 + 384*dlt + 128) + 160*dlt*E^3*( - 9*T_ex*dlt^5*lam*rho^2 - 54*T_ex*dlt^4*lam*rho^2 +
36*T_ex*dlt^3*lam*rho^2 + 396*T_ex*dlt^2*lam*rho^2 + 7*dlt^5*rho^2 + 27*dlt^4*rho^2 + 294*dlt^3*rho^2 +
316*dlt^2*rho^2 + 400*dlt^2 - 132*dlt*rho^2 - 96*dlt - 48) - E^6*(131*dlt^6*rho^2 + 1512*dlt^5*rho^2 -
2316*dlt^4*rho^2 - 125120*dlt^3*rho^2 - 28160*dlt^3 + 347520*dlt^2*rho^2 + 76800*dlt^2 - 299520*dlt*rho^2 -
65280*dlt + 57600*rho^2 + 11520) - 60*E^5*(576*T_ex^2*dlt^3*lam^2*rho^2 - 1152*T_ex^2*dlt^2*lam^2*rho^2 +
576*T_ex^2*dlt*lam^2*rho^2 - 6*T_ex*dlt^6*lam*rho^2 - 60*T_ex*dlt^5*lam*rho^2 + 144*T_ex*dlt^4*lam*rho^2 +
2016*T_ex*dlt^3*lam*rho^2 + 768*T_ex*dlt^3*lam - 5184*T_ex*dlt^2*lam*rho^2 - 1536*T_ex*dlt^2*lam +
3456*T_ex*dlt*lam*rho^2 + 768*T_ex*dlt*lam - 384*T_ex*lam*rho^2 + dlt^6*rho^2 - 116*dlt^4*rho^2 +
2336*dlt^3*rho^2 + 128*dlt^3 - 7008*dlt^2*rho^2 - 1024*dlt^2 + 5760*dlt*rho^2 + 1152*dlt - 1024*rho^2 -
256) - 15*E^4*(36*T_ex^2*dlt^6*lam^2*rho^2 + 288*T_ex^2*dlt^5*lam^2*rho^2 - 4608*T_ex^2*dlt^3*lam^2*rho^2 +
```

### 4.3 Two shortest chunks

#### Shortest #1 — id `7817a9939d-0016` — 98 tokens — `Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)`
Citation URL: https://arxiv.org/pdf/2502.07766

```text
Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)

Derivation of closed-form solutions without restrictions
Although useful as a simplification , th e restriction 𝛼 = 𝜃 when deriving the closed-form
expressions in the previous research (see Section 7) makes the model less suitable for the
calibration to volatility surfaces in equity markets, because of the market specifics mentioned
above (see Section 1).
```

#### Shortest #2 — id `7817a9939d-0028` — 127 tokens — `Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)`
Citation URL: https://arxiv.org/pdf/2502.07766

```text
Mean-Reverting SABR Models: Closed-form Surfaces and Calibration for Equities (arXiv PDF)

Appendix B: Closed-form expressions for mrSABR
The following formulas for mrSABR are in Excel format and use notations as follows: alp: parameter 𝛼 (initial instantaneous volatility)
the: parameter 𝜃 (long-term volatility)
lam: parameter 𝜆 (speed of mean reversion of volatility)
nu: parameter 𝜈 (lognormal volatility of volatility)
rho: parameter 𝜌 (correlation)
T_ex: maturity/expiry of option
E:
```

### 4.4 Ten most frequent repeated non-empty lines per ingested document

Body lines only (context header excluded); counts are occurrences across that document's chunks.

#### `https://arxiv.org/pdf/2502.07766` — 35 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 20 | `𝑇` |
| 2 | 8 | `𝑇1` |
| 3 | 8 | `𝑔` |
| 4 | 7 | ` 𝑑𝑇1` |
| 5 | 6 | `𝑇𝑒𝑥` |
| 6 | 4 | `𝑓 ≡ 𝐹0` |
| 7 | 4 | `with:` |
| 8 | 3 | `𝛼 ≡ 𝐴0` |
| 9 | 3 | `lambda rho alpha theta nu` |
| 10 | 3 | `Correlations: rho alpha theta nu` |

#### `https://bauer.uh.edu/spirrong/interest_rate_models.pdf` — 20 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 38 | `∫ T` |
| 2 | 5 | `∫ t` |
| 3 | 4 | `which implies:` |
| 4 | 4 | `A(T ) = .5σ2` |
| 5 | 3 | `P` |
| 6 | 3 | `and` |
| 7 | 3 | `ytdt = σ` |
| 8 | 3 | `∂T − aσ22a2 (1 − e−aT )e−aT` |
| 9 | 3 | `t` |
| 10 | 2 | `T − t` |

#### `https://csustan.csustan.edu/~tom/Lecture-Notes/HighFinance/` — 1 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 1 | `High Finance - Lecture Notes` |
| 2 | 1 | `Fall, 2008` |
| 3 | 1 | `These are some "lecture notes" and related material for a talk I gave recently.` |
| 4 | 1 | `The announcement for the talk looked like this:` |
| 5 | 1 | `Speaker: Dr. Tom Carter, Department of Computer Science, California State University, Stanislaus` |
| 6 | 1 | `Date: Tuesday, November 18, 2008` |
| 7 | 1 | `Time: 4:00 - 5:00 p.m.` |
| 8 | 1 | `Room: P104` |
| 9 | 1 | `Title: Fatal Shore -- Black-Scholes and High Finance` |
| 10 | 1 | `Abstract: This talk will include a derivation/analysis of the Black-Scholes` |

#### `https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/resources/mit15_450f10_lec02/` — 28 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 9 | `Outline` |
| 2 | 9 | `1 Stochastic Integral` |
| 3 | 9 | `2 Itô’s Lemma` |
| 4 | 9 | `3 Black-Scholes Model` |
| 5 | 9 | `4 Multivariate Itô Processes` |
| 6 | 9 | `5 SDEs` |
| 7 | 9 | `6 SDEs and PDEs` |
| 8 | 9 | `7 Risk-Neutral Probability` |
| 9 | 9 | `8 Risk-Neutral Pricing` |
| 10 | 8 | `t` |

#### `https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/resources/mit15_450f10_review01/` — 6 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 5 | `Problem` |
| 2 | 4 | `c` |
| 3 | 4 | `t` |
| 4 | 3 | `p` |
| 5 | 3 | `r` |
| 6 | 2 | `u` |
| 7 | 2 | `πt` |
| 8 | 2 | `Solution` |
| 9 | 1 | `Review: Arbitrage-Free Pricing and Stochastic` |
| 10 | 1 | `Calculus` |

#### `https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/resources/mit18_642_f24_lec19_1_pdf/` — 55 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 13 | `R t` |
| 2 | 13 | `R T` |
| 3 | 13 | `Computing Ito Integrals` |
| 4 | 11 | `Brownian Motion with Drift` |
| 5 | 10 | `Ito Process from Ito Integrals` |
| 6 | 10 | `Ito Integral: General Case R tXt (ω)= f (ω, s)dBs (ω), 0 ≤ t ≤ T0` |
| 7 | 9 | `I(f)(ω)=` |
| 8 | 9 | `Proof:` |
| 9 | 9 | `Squared-norm of f(ω, s)onΩ ×[t, t+ k]` |
| 10 | 9 | `f (ω, s) : function of ω and time s` |

#### `https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote17/` — 6 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 3 | `i` |
| 2 | 2 | `(Bti+1` |
| 3 | 1 | `Lecture 17 : Stochastic Processes II` |
| 4 | 1 | `1 Continuous-time stochastic process` |
| 5 | 1 | `So far we have studied discrete-time stochastic processes. We studied the` |
| 6 | 1 | `concept of Makov chains and martingales, time series analysis, and regression analysis on discrete-time stochastic processes.` |
| 7 | 1 | `We now turn our focus to the study of continuous-time stochastic processes. In most cases, it is diﬃcult to exactly describe the probability distribution for continuous-time stochastic processes. This was also diﬃcult` |
| 8 | 1 | `for discrete time stochastic processes, but for them, we described the distribution in terms of the increments Xk+1−Xk instead; this is impossible` |
| 9 | 1 | `for continuous time stochastic processes. An alternate way which is commonly used is to ﬁrst describe the properties satisﬁed by the probability` |
| 10 | 1 | `distribution, and then to show that there exists a probability distribution` |

#### `https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote21/` — 6 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 2 | `ˆ T` |
| 2 | 2 | `then` |
| 3 | 2 | `df(t,Bt) =g(t,Bt)dBt +h(t,Bt)dt.` |
| 4 | 2 | `ˆ ∞` |
| 5 | 1 | `Lecture 21: Stochastic Differential Equations` |
| 6 | 1 | `In this lecture, we study stochastic diﬀerential equations. See Chapter 9` |
| 7 | 1 | `of [3] for a thorough treatment of the materials in this section.` |
| 8 | 1 | `1. Stochastic differential equations` |
| 9 | 1 | `We would like to solve diﬀerential equations of the form` |
| 10 | 1 | `dX =µ(t,X (t ))dt +σ(t, X (t))dB(t )` |

#### `https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote5/` — 7 chunks

| rank | count | exact line |
|---|---|---|
| 1 | 3 | `E` |
| 2 | 2 | `k=1` |
| 3 | 2 | `T−1` |
| 4 | 2 | `= E` |
| 5 | 1 | `Lecture 5 : Stochastic Processes I` |
| 6 | 1 | `1 Stochastic process` |
| 7 | 1 | `A stochastic process is a collection of random variables indexed by time.` |
| 8 | 1 | `An alternate view is that it is a probability distribution over a space` |
| 9 | 1 | `of paths; this path often describes the evolution of some random value, or` |
| 10 | 1 | `system, over time. In a deterministic process, there is a ﬁxed trajectory` |

## 5. Test evidence

Commands run after the last code change (`tests/test_ingest_pipeline.py`, 22:17:28 local).

```text
$ date -u +%Y-%m-%dT%H:%M:%SZ
2026-10-23T19:42:52Z

$ uv run pytest --collect-only -q | grep -i ingest
tests/test_ingest_chunking.py: 6
tests/test_ingest_extract.py: 2
tests/test_ingest_fetch.py: 5
tests/test_ingest_pipeline.py: 5
tests/test_ingest_sources.py: 5

$ uv run pytest -q
........................................................................ [ 60%]
...............................................s                         [100%]
(exit 0; addopts already contains -q, so -q -q suppresses the counts line)

$ uv run pytest
........................................................................ [ 60%]
...............................................s                         [100%]
119 passed, 1 skipped in 7.43s

$ uv run pytest -q -rs
........................................................................ [ 60%]
...............................................s                         [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_smoke_openrouter.py:47: set QUANTRAG_SMOKE=1 to run the live OpenRouter test

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
33 files already formatted
```

`uv run pytest --collect-only -q` reports per-module counts (because `addopts` already contains `-q`); the two required regression tests are shown as node ids with `addopts` cleared:

```text
$ uv run pytest --collect-only -o addopts="" -q | grep -E "dry_run_does_not_clobber|headers_page_numbers"
tests/test_ingest_extract.py::test_extract_pdf_headers_page_numbers_logic
tests/test_ingest_pipeline.py::test_dry_run_does_not_clobber_real_report
```

The only skip is `tests/test_smoke_openrouter.py:47` (`set QUANTRAG_SMOKE=1 to run the live OpenRouter test`).

## 6. Artifact provenance and cross-checks

| artifact | bytes | mtime (local +03:00) | sha256 |
|---|---|---|---|
| `data/chunks.jsonl` | 275387 | 2026-10-23 22:42:49 | `057085f406f1fb27a0a739a9b016084ef67bf0f63bd99db25a49ad0ce9ba7d07` |
| `data/ingest_report.json` | 46630 | 2026-10-23 22:42:49 | `316d2620316b25ba247d243e2c3ab36109e25439e59d84f64b728bb5bc223885` |
| `data/ingest_report.csv` | 23806 | 2026-10-23 22:42:49 | `9d1c9743e55f6c3d860a2e4d0dfe7c172a386d8d75b11d2bbb12e03681776541` |
| `data/ingest_report.dryrun.json` | 43449 | 2026-10-23 22:17:17 | `58647e09a34b0522d92816edc4fdbb1c6ca6019b908960a4a558732363590105` |
| `data/ingest_report.dryrun.csv` | 21637 | 2026-10-23 22:17:17 | `cfdb17bb7bc6a45d2d7d04c1833ffdd0898c3a0555c2d1e340ec49eacb28e63f` |
| `docs/phase1_audit.md` | 32885 | generated from the above | `docs/phase1_audit.md.sha256` |

- Real pilot artifacts were written by the same process invocation (22:42:49), in this order: chunks, report.json, report.csv. Re-running the identical command produced byte-identical hashes (determinism check).
- `data/ingest_report.csv` has 121 lines = 1 header + 120 rows.
- `data/ingest_report.json` totals equal the recomputation from `data/chunks.jsonl` (164 / 494.81 / 482 / 878 / 10 / 0) and the per-source chunk sum (164).
- The dry-run files are from a separate `--dry-run --profile all` invocation at 22:17:17, so they intentionally contain `selected`/`skipped` rows and empty chunk columns; they reconcile as 63 selected + 57 skipped = 120. The `--profile core` dry-run (24 + 96 = 120) was printed to stdout earlier; its file was overwritten because both invocations share one filename. That is a reporting limitation, not a pilot discrepancy.
- The page-furniture and dry-run regressions are in the passing suite (section 5).

## 7. PASS / FAIL / BLOCKED checklist

| item | verdict | evidence |
|---|---|---|
| Scope separation (manifest vs profile vs pilot) | PASS | §0, §1 |
| Ten pilot rows with status/reason/chunks/pages/chars/original/used URL | PASS | §1 |
| Aggregates recomputed from rows + artifacts | PASS | §1, §6 |
| Token stats incl. >800/>900/>1000, all ≤900 | PASS | §2 |
| Seven-failure diagnostic (unavailable metadata marked) | PASS | §3 |
| Five verbatim chunks / longest 2 / shortest 2 / repeated lines | PASS | §4 |
| Test commands + skipped test + run after last change | PASS | §5 |
| Artifact existence/hash/same-run agreement | PASS | §6 |
| Dry-run files agree with dry-run totals | PASS (core file overwritten by all-profile run — stated) | §6 |
| robots.txt compliance | PASS | row 7 not fetched |
| Phase 2 untouched | PASS | no `rag.py` / index / MCP changes |
