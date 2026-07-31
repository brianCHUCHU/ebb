# TSB-HB v2 — Formal Theory (LaTeX-ready)

> **狀態（2026-07-29）**：本檔涵蓋 Prop 1–3 + Cor 3.1。論文另有 **Prop 4**（selector 的
> oracle 不等式，`prop:oracle`）與 **Prop 5 候選**（credibility 權重的雙邊界，
> 見 `paper_v2/STATUS.md` §6 P-3），兩者只存在於 `paper_v2/main.tex`／待撰，尚未回寫到本檔。
> 若本檔與 `main.tex` 或 `tests/test_theory_symbolic.py` 不一致，以後兩者為準。

Target: Section 3.x "Theoretical properties" of the AISTATS 2027 submission.
All statements are proved in full below; proofs are elementary but the
structure (exact TSB recovery + always-active shrinkage + drift-tracking
rates) is the formal backbone of the "unification" claim.

Notation. Fix item $i$ (index suppressed). The initialization window is
$t = 1,\dots,T$. Occurrence indicators $p_t = \mathbb{1}\{y_t > 0\}$.
For a discount $w \in (0,1]$ define the discounted sufficient statistics

$$
n(w) \;=\; \sum_{t=1}^{T} w^{T-t}, \qquad
s(w) \;=\; \sum_{t=1}^{T} w^{T-t}\, p_t ,
$$

and the discounted empirical rate $\bar p(w) = s(w)/n(w)$ (defined when
$n(w)>0$). Under the group prior $\pi \sim \mathrm{Beta}(\alpha,\beta)$ the
weighted-likelihood (power-prior) posterior mean used by TSB-HB v2 is

$$
\hat\pi(w;\alpha,\beta) \;=\; \frac{\alpha + s(w)}{\alpha + \beta + n(w)}.
\tag{1}
$$

Recall classical TSB updates the occurrence estimate by exponential
smoothing with parameter $\alpha_p \in (0,1)$:
$\hat p_t = \hat p_{t-1} + \alpha_p (p_t - \hat p_{t-1})$, initialized at
some $\hat p_0 \in [0,1]$.

---

## Proposition 1 (Classical TSB is the noninformative-prior limit)

Let $w \in (0,1)$ and set $\alpha_p = 1-w$. Let $\hat p_T$ denote the
classical TSB occurrence estimate after processing $p_1,\dots,p_T$ from
initialization $\hat p_0$. Then:

**(i)** $\displaystyle \lim_{(\alpha,\beta)\to(0,0)} \hat\pi(w;\alpha,\beta) = \bar p(w).$

**(ii)** The TSB recursion has the closed form
$$
\hat p_T \;=\; (1-w) \sum_{t=1}^{T} w^{T-t} p_t \;+\; w^{T} \hat p_0 ,
$$
and consequently
$$
\bar p(w) \;=\; \frac{\hat p_T - w^{T}\hat p_0}{1 - w^{T}} .
$$

**(iii)** Hence $\bar p(w) = \hat p_T$ **exactly** when the recursion is
initialized self-consistently at $\hat p_0 = \bar p(w)$, and for any
initialization,
$$
\bigl|\, \bar p(w) - \hat p_T \,\bigr| \;=\; \frac{w^{T}\,(1-w^{T})^{-1}\cdot 0 + w^T\,|\bar p(w) - \hat p_0|\,}{1} \;\le\; w^{T}\longrightarrow 0
$$
geometrically as $T \to \infty$.

**Proof.**
(i) Immediate from (1) since $s(w), n(w)$ are fixed and finite.
(ii) Induction on $T$: for $T=1$, $\hat p_1 = (1-w)p_1 + w\hat p_0$. If the
form holds at $T-1$ then
$\hat p_T = (1-w)p_T + w\hat p_{T-1}
= (1-w)p_T + w\left[(1-w)\sum_{t=1}^{T-1} w^{T-1-t}p_t + w^{T-1}\hat p_0\right]$,
which is the stated form. Since $n(w) = \sum_{k=0}^{T-1} w^k = (1-w^{T})/(1-w)$,
$$
\bar p(w) = \frac{(1-w)\sum_t w^{T-t}p_t}{1-w^{T}}
= \frac{\hat p_T - w^{T}\hat p_0}{1-w^{T}} .
$$
(iii) Rearranging, $\bar p(w) - \hat p_T = w^{T}\bigl(\bar p(w) - \hat p_0\bigr)$,
and $|\bar p(w) - \hat p_0| \le 1$. $\square$

**Remark 1 (size block).** The analogous statement holds for the size
block: as the shrinkage ratio $k = \sigma^2_{\mathrm{proc}}/\tau^2 \to 0$,
the posterior log-size mean $\hat\mu$ converges to the discounted average of
observed log-sizes, i.e., an exponentially weighted *geometric* mean of
sizes; the plug-in forecast $\exp(\hat\mu + \tfrac12\hat\sigma^2)$ is its
log-normal mean correction. Classical TSB smooths sizes arithmetically;
under the log-normal size model both are estimators of the same conditional
mean $\mathbb{E}[S\mid S>0]$.

**Interpretation.** TSB-HB v2 spans a two-dimensional model plane with axes
*shrinkage strength* $\phi = \alpha+\beta$ (cross-sectional pooling) and
*discount* $w$ (temporal forgetting). Classical TSB is the edge
$\phi \to 0$; released TSB-HB v1 is the edge $w = 1$. v2 selects both
coordinates from data.

---

## Proposition 2 (Discounting keeps hierarchical shrinkage active)

Write (1) as a credibility combination
$$
\hat\pi(w) \;=\; \lambda(w)\,\bar p(w) + \bigl(1-\lambda(w)\bigr)\,\mu_\pi,
\qquad
\lambda(w) = \frac{n(w)}{n(w) + \phi}, \quad
\mu_\pi = \frac{\alpha}{\alpha+\beta}, \quad \phi = \alpha + \beta .
$$
Then for every horizon $T$,
$$
\lambda(w) \;\le\; \frac{1}{1 + (1-w)\,\phi} \;<\; 1 \qquad (w < 1),
$$
whereas for $w = 1$, $\lambda(1) = T/(T+\phi) \to 1$ as $T \to \infty$.

**Proof.** $n(w) = (1-w^{T})/(1-w) \le 1/(1-w)$, and
$x \mapsto x/(x+\phi)$ is increasing. $\square$

**Interpretation.** With forgetting, the effective sample size is bounded by
$1/(1-w)$, so the pooled prior retains a strictly positive weight *forever*:
temporal adaptation and cross-sectional pooling are complements, not
substitutes. Classical TSB ($\phi\to0$) discards exactly the information
that remains valuable under forgetting. This is the formal answer to "why
hierarchical TSB with discounting rather than plain TSB."

---

## Proposition 3 (Finite-sample MSE under occurrence drift)

Assume $p_t \sim \mathrm{Bernoulli}(\pi_t)$ independently, with Lipschitz
drift toward the forecast origin:
$|\pi_t - \pi_T| \le \delta\,(T-t)$ for all $t \le T$ and some
$\delta \ge 0$. Let $b = 1-w \in (0,1)$. Then the discounted rate satisfies

$$
\bigl|\mathbb{E}\,\bar p(w) - \pi_T\bigr|
\;\le\; \delta\, \frac{w}{(1-w)\,(1-w^{T})},
\qquad
\mathrm{Var}\,\bar p(w)
\;\le\; \frac{1}{4}\,\frac{(1-w)\,(1+w^{T})}{(1+w)\,(1-w^{T})},
$$

and the posterior mean $\hat\pi(w) = \lambda \bar p(w) + (1-\lambda)\mu_\pi$
obeys
$$
\mathbb{E}\bigl(\hat\pi(w) - \pi_T\bigr)^2
\;\le\;
\Bigl[\lambda\,\mathrm{B}(w) + (1-\lambda)\,|\mu_\pi - \pi_T|\Bigr]^2
+ \lambda^2\,\mathrm{Var}\,\bar p(w),
$$
where $\mathrm{B}(w)$ is the bias bound above.

**Proof.** Bias:
$\mathbb{E}\bar p(w) - \pi_T = \sum_t w^{T-t}(\pi_t - \pi_T)/n(w)$, so with
$k = T-t$,
$$
\bigl|\mathbb{E}\bar p(w) - \pi_T\bigr|
\le \frac{\delta \sum_{k=0}^{T-1} k\,w^{k}}{n(w)}
\le \frac{\delta\, w/(1-w)^2}{(1-w^{T})/(1-w)}
= \delta\,\frac{w}{(1-w)(1-w^{T})},
$$
using $\sum_{k\ge0} k w^k = w/(1-w)^2$. Variance: by independence and
$\pi_t(1-\pi_t)\le \tfrac14$,
$$
\mathrm{Var}\,\bar p(w) \le \frac{\tfrac14 \sum_{k=0}^{T-1} w^{2k}}{n(w)^2}
= \frac{1}{4}\,
\frac{(1-w^{2T})/(1-w^2)}{\bigl[(1-w^{T})/(1-w)\bigr]^2}
= \frac{1}{4}\,\frac{(1-w)(1+w^{T})}{(1+w)(1-w^{T})}.
$$
The posterior-mean bound follows from the decomposition
$\hat\pi - \pi_T = \lambda(\bar p - \mathbb{E}\bar p) + \lambda(\mathbb{E}\bar p - \pi_T) + (1-\lambda)(\mu_\pi - \pi_T)$
and $(a+b)^2 \le$ (bias sum)$^2$ + variance term with the zero-mean first
component. $\square$

**Corollary 3.1 (optimal discount rate).** Write $b = 1-w$. As $T \to \infty$
the variance bound above tends to $\tfrac{1-w}{4(1+w)} = b/8 + O(b^2)$, so for
small drift $\delta$ the trade-off to balance is $\delta^2/b^2$ against $b/8$.
Minimizing $\delta^2/b^2 + cb$ gives $b^\* = (2\delta^2/c)^{1/3}$, hence with
$c = 1/8$

$$
b^\* = 1-w^\* = (16\delta^2)^{1/3} = 2^{4/3}\,\delta^{2/3},
\qquad \mathrm{MSE} = O(\delta^{2/3}).
$$

In particular $w^\* \to 1$ as $\delta \to 0$: the stationary released model is
recovered exactly when there is no drift, which is the formal justification for
*selecting* $w$ on a chronological validation split rather than fixing it.

> **Corrected 2026-07-25 (task D1).** An earlier draft of this note balanced
> against $b/4$ and reported $b^\* \asymp (2\delta)^{2/3}$; those two constants
> are mutually inconsistent and neither matches the $b/8$ limit of the variance
> bound. The $O(\delta^{2/3})$ *rate* is unaffected. `paper_v2/main.tex` states
> the corollary at rate level and carries the exact constant in the Appendix A.3
> proof; `tests/test_theory_symbolic.py` verifies all three propositions with
> `sympy` and is the authority if this note and the paper ever disagree again.

**Remark 2 (consistency is preserved).** With $w = 1$ all statements reduce
to the released v1 model and its consistency results (supplement of v1)
apply unchanged; with $w < 1$, Proposition 2 shows the estimator tracks a
bounded-effective-sample process, which is the correct behavior under
nonstationarity — consistency for a *fixed* $\pi$ is deliberately traded
for $O(\delta^{2/3})$ tracking risk under drift.

**Remark 3 (mixture pooling, BIC).** For the learned partition, each EM
iteration maximizes a valid lower bound of the mixture marginal likelihood
built from the same Beta-Binomial and Normal random-effects marginals
(E-step exact by conjugacy of the component marginals; M-step =
responsibility-weighted versions of the v1 EB objectives). Model-selection
consistency of BIC for finite mixtures under standard regularity conditions
follows Keribin (2000); we use BIC to select $K$ and treat this as a
citation-level guarantee, not a new result.

---

## How this lands the review criticisms

- **bm76 (no temporal adaptation):** Prop 1 + Cor 3.1 — the model now
  *contains* TSB's forgetting mechanism and selects its strength from data.
- **KGCf / zsGB (incremental novelty):** the contribution is no longer
  "components glued together": Prop 2 is a structural statement about why
  pooling and forgetting are complements; the model plane
  $(\phi, w)$ with data-driven selection of both coordinates is the paper's
  thesis.
- **v1's "theory as decoration":** every statement above is used in the
  narrative (TSB recovery → positioning; always-active shrinkage → why HB;
  drift MSE → why/how to choose $w$).
