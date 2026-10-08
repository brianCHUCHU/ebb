"""Symbolic verification of Propositions 1-3 and Corollary 1 (paper Sec. 3.5).

Run: py -m pytest tests/test_theory_symbolic.py  (or plain: py tests/test_theory_symbolic.py)
"""

import sympy as sp


def test_prop1_tsb_closed_form_and_identity():
    w, p0 = sp.symbols("w p0", positive=True)
    T = 6
    p = sp.symbols(f"p1:{T+1}")
    # TSB recursion
    phat = p0
    for t in range(T):
        phat = phat + (1 - w) * (p[t] - phat)
    phat = sp.expand(phat)
    closed = sp.expand((1 - w) * sum(w ** (T - 1 - t) * p[t] for t in range(T)) + w**T * p0)
    assert sp.simplify(phat - closed) == 0
    # discounted mean identity: pbar = (phat - w^T p0) / (1 - w^T)
    n_w = sum(w ** (T - 1 - t) for t in range(T))
    s_w = sum(w ** (T - 1 - t) * p[t] for t in range(T))
    pbar = s_w / n_w
    ident = (phat - w**T * p0) / (1 - w**T)
    assert sp.simplify(sp.together(pbar - ident)) == 0
    # difference formula: pbar - phat = w^T (pbar - p0)
    assert sp.simplify(pbar - phat - w**T * (pbar - p0)) == 0


def test_prop2_lambda_bound():
    w, phi, T = sp.symbols("w phi T", positive=True)
    n_w = (1 - w**T) / (1 - w)
    lam = n_w / (n_w + phi)
    bound = 1 / (1 + (1 - w) * phi)
    # lambda <= bound  <=>  bound - lambda >= 0 for 0<w<1, T>0, phi>0
    diff = sp.simplify(bound - lam)
    for wv, Tv, phiv in [(sp.Rational(1, 2), 5, 3), (sp.Rational(95, 100), 200, 10), (sp.Rational(999, 1000), 50, 1)]:
        assert diff.subs({w: wv, T: Tv, phi: phiv}) >= 0


def test_prop3_geometric_sums_and_variance_constant():
    w = sp.symbols("w", positive=True)
    k, T = sp.symbols("k T", positive=True, integer=True)
    # bias numerator: partial sums of k w^k are bounded by w/(1-w)^2 for 0<w<1
    for wv in [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(9, 10)]:
        partial = sum(kk * wv**kk for kk in range(400))
        assert partial <= wv / (1 - wv) ** 2
        # and converges to it (within tolerance at high truncation)
        assert sp.Abs(partial - wv / (1 - wv) ** 2) < sp.Rational(1, 10**6)
    # variance ratio: (sum w^{2k}) / (sum w^k)^2 over k=0..T-1
    num = (1 - w ** (2 * T)) / (1 - w**2)
    den = ((1 - w**T) / (1 - w)) ** 2
    ratio = sp.simplify(num / den)
    target = (1 - w) * (1 + w**T) / ((1 + w) * (1 - w**T))
    assert sp.simplify(ratio - target) == 0
    # T -> oo limit equals (1-w)/(1+w); with b=1-w this is b/(2-b); x 1/4 -> b/8 leading order
    b = sp.symbols("b", positive=True)
    for bv in [sp.Rational(1, 10), sp.Rational(1, 2)]:
        lim = sp.limit(target.subs(w, 1 - bv), T, sp.oo)
        assert sp.simplify(lim - bv / (2 - bv)) == 0
    assert sp.limit((sp.Rational(1, 4) * b / (2 - b)) / (b / 8), b, 0) == 1


def test_corollary_constant():
    b, delta = sp.symbols("b delta", positive=True)
    f = delta**2 / b**2 + b / 8
    bstar = sp.solve(sp.diff(f, b), b)
    real = [s for s in bstar if s.is_real]
    assert sp.simplify(real[0] - (16 * delta**2) ** sp.Rational(1, 3)) == 0


def test_prop5_upper_bound_size_block():
    """Prop 5(i): the size credibility weight is bounded away from 1 under forgetting."""
    w, kappa, T = sp.symbols("w kappa T", positive=True)
    n_w = (1 - w**T) / (1 - w)
    lam = n_w / (n_w + kappa)
    bound = 1 / (1 + (1 - w) * kappa)
    diff = sp.simplify(bound - lam)
    for wv, Tv, kv in [
        (sp.Rational(1, 2), 5, 3),
        (sp.Rational(98, 100), 120, sp.Rational(1, 2)),
        (sp.Rational(9, 10), 120, 10),
    ]:
        assert diff.subs({w: wv, T: Tv, kappa: kv}) >= 0
    # strictly interior: the bound is < 1 whenever w < 1
    assert sp.simplify(bound.subs({w: sp.Rational(9, 10), kappa: 2})) < 1


def test_prop5_moment_estimator_is_unbiased_before_truncation():
    """Prop 5(ii): E[S_g^2] = tau^2 + mean(s_i) exactly, for unequal sampling variances."""
    m = 4
    tau2 = sp.symbols("tau2", positive=True)
    s = sp.symbols(f"s1:{m+1}", positive=True)
    mu = sp.symbols(f"mu1:{m+1}")
    # Var(ybar_i) = tau2 + s_i, independent across i.
    var_y = [tau2 + s[i] for i in range(m)]
    # E[S^2] = (1/(m-1)) * ( sum Var(y_i) - m Var(ybar) ), Var(ybar) = (1/m^2) sum Var(y_i)
    total = sum(var_y)
    e_s2 = sp.simplify((total - total / m) / (m - 1))
    assert sp.simplify(e_s2 - (tau2 + sum(s) / m)) == 0


def test_prop5_lower_bound_under_credibility_regularizer():
    """Prop 5(iii): with omega < 1 the regularized weight is bounded away from 0."""
    n, sigma2, tau0, omega = sp.symbols("n sigma2 tau0 omega", positive=True)
    tau_reg = (1 - omega) * tau0  # lower bound on omega*tauhat^2 + (1-omega)*tau0^2
    lam_lower = n / (n + sigma2 / tau_reg)
    closed = n * (1 - omega) * tau0 / (n * (1 - omega) * tau0 + sigma2)
    assert sp.simplify(lam_lower - closed) == 0
    # strictly positive for any finite kappa
    val = closed.subs({n: sp.Rational(1, 2), sigma2: 10, tau0: sp.Rational(1, 100), omega: sp.Rational(9, 10)})
    assert val > 0
    # monotone: more regularization (smaller omega) raises the floor
    d_omega = sp.simplify(sp.diff(closed, omega))
    assert d_omega.subs({n: 5, sigma2: 1, tau0: 1, omega: sp.Rational(1, 2)}) < 0


def test_prop5_forgetting_lowers_the_identifiability_ratio():
    """Prop 5(ii): rho_g = tau^2/(tau^2 + sbar) is decreasing in the discount gap (1-w)."""
    tau2, sigma2, w = sp.symbols("tau2 sigma2 w", positive=True)
    sbar_lower = (1 - w) * sigma2  # since n_i <= 1/(1-w)
    rho_upper = tau2 / (tau2 + sbar_lower)
    # d rho / d w > 0  <=>  rho decreases as w decreases (more forgetting)
    assert sp.simplify(sp.diff(rho_upper, w)).subs({tau2: 1, sigma2: 2, w: sp.Rational(9, 10)}) > 0


def test_prop5_collapse_probability_matches_simulation():
    """Prop 5(ii): the Gaussian approximation to Pr[tauhat^2 = 0] tracks a Monte Carlo run."""
    import numpy as np
    from scipy.stats import norm

    rng = np.random.default_rng(20260729)
    for m_g, tau2, s in [(50, 0.05, 0.5), (200, 0.02, 0.4), (30, 0.30, 0.3)]:
        draws = rng.normal(0.0, np.sqrt(tau2 + s), size=(20000, m_g))
        s2 = draws.var(axis=1, ddof=1)
        empirical = float(np.mean(s2 - s <= 0.0))
        rho = tau2 / (tau2 + s)
        predicted = float(norm.cdf(-np.sqrt((m_g - 1) / 2.0) * rho))
        assert abs(empirical - predicted) < 0.05, (m_g, tau2, s, empirical, predicted)

def test_lemma_noise_floor_endpoint_and_monotonicity():
    """Lemma (noise floor): g(r) = (1 + c r^2)/(1 + r)^2 is decreasing for r < 1/c and
    equals c = (1-w)/(1+w) at r = w/(1-w)."""
    w, r = sp.symbols("w r", positive=True)
    c = (1 - w) / (1 + w)
    g = (1 + c * r**2) / (1 + r) ** 2
    assert sp.simplify(g.subs(r, w / (1 - w)) - c) == 0
    assert sp.simplify(sp.diff(g, r) - 2 * (c * r - 1) / (1 + r) ** 3) == 0
    assert sp.simplify(1 / c - w / (1 - w) - 1 / (1 - w)) == 0   # 1/c - w/(1-w) = 1/(1-w) > 0


def test_lemma_noise_floor_numeric():
    """Lemma (noise floor): sum a^2 / (sum a)^2 lies in [(1-w)/(1+w), 1/sum a] for weights
    a_j = w^{k_j} with arbitrary distinct non-negative integer exponents."""
    import numpy as np

    rng = np.random.default_rng(20260930)
    for _ in range(20000):
        wv = float(rng.uniform(0.01, 0.999))
        n = int(rng.integers(1, 60))
        k = np.sort(rng.choice(400, size=n, replace=False))
        # the ratio is invariant to rescaling the weights; rescale so that the largest weight is 1
        # (as in the proof), which also avoids floating-point underflow for large exponents
        a = wv ** (k - k.min()).astype(float)
        ratio = float(np.sum(a**2) / np.sum(a) ** 2)
        assert ratio >= (1 - wv) / (1 + wv) - 1e-12, (wv, k, ratio)
        # upper bound: sum a^2 <= sum a for weights in (0, 1]; the unrescaled bound 1/sum(a) is weaker
        assert ratio <= 1.0 / float(np.sum(a)) + 1e-12, (wv, k, ratio)
    # the bound is attained in the limit of a full geometric history
    wv = 0.9
    a = wv ** np.arange(5000.0)
    assert abs(np.sum(a**2) / np.sum(a) ** 2 - (1 - wv) / (1 + wv)) < 1e-9


def test_prop5_variance_of_sample_variance_under_imbalance():
    """Prop (ii), second moment: tr(MVMV) = (1 - 2/m) sum v^2 + (sum v)^2/m^2 >= (m-1) vbar^2,
    with equality iff the v_i are equal; checked symbolically (m = 4) and by simulation."""
    import numpy as np

    m = 4
    v = sp.symbols(f"v1:{m+1}", positive=True)
    M = sp.eye(m) - sp.ones(m, m) / m
    V = sp.diag(*v)
    tr = sp.expand((M * V * M * V).trace())
    closed = sp.expand((1 - sp.Rational(2, m)) * sum(x**2 for x in v) + sum(v) ** 2 / m**2)
    assert sp.simplify(tr - closed) == 0
    vbar = sum(v) / m
    gap = sp.expand(closed - (m - 1) * vbar**2)
    # the gap is (1 - 2/m) * sum (v_i - vbar)^2
    assert sp.simplify(gap - (1 - sp.Rational(2, m)) * sum((x - vbar) ** 2 for x in v)) == 0
    assert gap.subs({v[0]: 1, v[1]: 1, v[2]: 1, v[3]: 1}) == 0
    # simulation: Var(S^2) matches 2 tr(MVMV)/(m-1)^2 and exceeds the balanced value
    rng = np.random.default_rng(7)
    vv = np.array([0.2, 0.5, 1.0, 3.0, 0.3, 0.9])
    mm = len(vv)
    y = rng.normal(0.0, np.sqrt(vv), size=(400000, mm))
    s2 = y.var(axis=1, ddof=1)
    Mn = np.eye(mm) - np.ones((mm, mm)) / mm
    exact = 2 * np.trace(Mn @ np.diag(vv) @ Mn @ np.diag(vv)) / (mm - 1) ** 2
    assert abs(s2.var() / exact - 1) < 0.03
    assert exact >= 2 * vv.mean() ** 2 / (mm - 1)


def test_necessary_direction_under_exact_sampling_variance():
    """(tau^2 - d)/(tau^2 + sbar - d) is decreasing in d and equals rho at d = 0."""
    tau2, sbar, d = sp.symbols("tau2 sbar d", positive=True)
    ratio = (tau2 - d) / (tau2 + sbar - d)
    assert sp.simplify(ratio.subs(d, 0) - tau2 / (tau2 + sbar)) == 0
    assert sp.simplify(sp.diff(ratio, d) + sbar / (tau2 + sbar - d) ** 2) == 0


def test_corollary_threshold_equivalence():
    """rho >= a  <=>  tau^2 >= a*sbar/(1-a) for 0 < a < 1; no solution for a >= 1."""
    tau2, sbar, a = sp.symbols("tau2 sbar a", positive=True)
    rho = tau2 / (tau2 + sbar)
    thr = a * sbar / (1 - a)
    assert sp.simplify(rho.subs(tau2, thr) - a) == 0
    assert sp.diff(rho, tau2).subs({tau2: 1, sbar: 1}) > 0
    assert sp.limit(rho, tau2, sp.oo) == 1


def test_weighted_hoeffding_effective_sample_size():
    """n_eff = 1/sum(omega^2) equals the number of items under equal weights and is smaller otherwise."""
    import numpy as np

    assert abs(1 / np.sum(np.full(50, 1 / 50) ** 2) - 50) < 1e-9
    om = np.array([0.5, 0.3, 0.1, 0.1])
    assert 1 / np.sum(om**2) < len(om)


if __name__ == "__main__":
    test_prop1_tsb_closed_form_and_identity()
    test_prop2_lambda_bound()
    test_prop3_geometric_sums_and_variance_constant()
    test_corollary_constant()
    test_prop5_upper_bound_size_block()
    test_prop5_moment_estimator_is_unbiased_before_truncation()
    test_prop5_lower_bound_under_credibility_regularizer()
    test_prop5_forgetting_lowers_the_identifiability_ratio()
    test_prop5_collapse_probability_matches_simulation()
    test_lemma_noise_floor_endpoint_and_monotonicity()
    test_lemma_noise_floor_numeric()
    test_prop5_variance_of_sample_variance_under_imbalance()
    test_necessary_direction_under_exact_sampling_variance()
    test_corollary_threshold_equivalence()
    test_weighted_hoeffding_effective_sample_size()
    print("all symbolic checks passed")
