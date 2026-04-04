# Seq Features:


6F_gamma+iv_lag:  [delta, T, spy_ret, vix_lag, vix_mom_lag, gamma, iv_lag],
8F_rho+iv_lag:    [delta, T, spy_ret, vix_lag, vix_mom_lag, vix_mom, gamma, rho, iv_lag],
8F_theta+iv_lag:  [delta, T, spy_ret, vix_lag, vix_mom_lag, vix_mom, gamma, theta, iv_lag],

- 3F:       (Basline): delta, T, spy ret
- 6F:       3F + vix_lag, vix_mom_lag, gamma, iv_lag
- 8F rho:   6F + vix_mom, rho
- 8F theta: 6F + vix_mom, theta


########################################

### The Short Answer
**It's not classical overfitting, but the result is period-dependent and needs context before you can trust it fully.**

---

### Why it's NOT overfitting

1. **The test set is genuinely future data** — chronological splits mean the model never saw the test period during training. R²=0.92 on truly unseen future data is not memorisation.

2. **All three feature sets cluster tightly** (91.7%, 91.2%, 91.0%) — if it were overfitting you'd expect wild variance across configurations, not this kind of stability.

3. **The residual diagnostics looked clean** — std dropped from 0.0122 → 0.0033, skew went from 10.86 → −0.13. A severely overfit model typically shows systematic residual patterns, not near-symmetric well-behaved errors.

4. **The absolute SSE is genuinely lower** (1.79 vs chro-A's 5.33) — the model is making tighter predictions in absolute terms, not just benefiting from a favourable baseline.

---

### Why you should be cautious

1. **The analytic baseline is much weaker on chro-B** (SSE=21.68) vs chro-A (SSE=14.07) — a weaker baseline inflates the % gain. The 91.7% partially reflects the analytic model struggling on this period, not just the LSTM excelling.

2. **chro-A vs chro-B inconsistency** — jumping from +62% to +92% across splits is a large swing. It strongly suggests the result is **regime-specific**: chro-B's test period likely has a market structure where iv_lag is an exceptionally strong signal (e.g., a trending or low-noise volatility period).

3. **You only have one data point for chro-B** — you need C and D to know if 91% is a fluke or a pattern.

---

### What to Watch For

| Scenario | Interpretation |
|----------|---------------|
| chro-C and D also ~85-92% | Results are robust across periods — very strong paper finding |
| chro-C and D drop back to ~60% | chro-B was a lucky period — report the average and variance |
| GRU chro-B also hits ~90% | Validates the result across architectures — even stronger |
| GRU chro-B is much lower | Architecture-specific fluke — worth investigating |

---

### For Your Paper
Report it as-is but **don't lead with chro-B in isolation**. Once you have all four splits, report the mean and standard deviation of gain across splits. If the average is, say, 75% ± 15%, that's a legitimate and strong result. If chro-B alone is pulling the average up, flag it explicitly as period-dependent performance.

The most important next step is getting chro-C and chro-D results.

######################################################


The core intuition
The classic "volatility smirk" relationship is being visualized here. When the market falls (blue), implied volatility rises across the surface (positive z-values). When the market rises (red), IV falls (negative z-values). This is the well-known negative correlation between equity returns and volatility.
The analytic model is linear in the return, so the impact of a +X% gain is equal and opposite to a -X% gain. You can see this in the top row: the blue and red surfaces are almost perfect mirror images of each other, smooth and well-behaved.
The key difference: ANN bottom row
The bottom row (your 3F ANN) tells a more interesting story. The surfaces are wavy and non-symmetric. The reduction in implied volatilities from a +1.25% daily return is on average about twice as large as the increase from a -1.25% return. This suggests the analytic model's predicted change is too high for large negative returns and too low for large positive returns.
The ANN has learned this asymmetry from the data, which the linear analytic model structurally cannot capture.


###########################################

