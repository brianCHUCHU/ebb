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


if __name__ == "__main__":
    test_prop1_tsb_closed_form_and_identity()
    test_prop2_lambda_bound()
    test_prop3_geometric_sums_and_variance_constant()
    test_corollary_constant()
    print("all symbolic checks passed")
