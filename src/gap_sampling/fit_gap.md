# The calibration procedure of fit_gap.py

Code: `fit_gap.py` (driver), `_calibration.py` (the fits and the map), `_sorted_samples.py` (sorting and aggregation).

## 0. Summary

The script estimates from samples a map $\Lambda(g)$ that converts the complementary gap $g$ produced by the inner decoder into the log-likelihood ratio (llr) used by the outer decoder. It has four steps.

1. Sort the samples by gap and count, for every gap value, the number of samples and the number of mismatches.
2. **Affine fit**: maximum-likelihood estimate of $\Lambda_{\mathrm{aff}}(g) = s\,g + o$ by logistic regression.
3. **Piecewise fit**: a per-bin correction to the affine fit, giving a monotone piecewise-linear map $\Lambda_{\mathrm{pw}}(g)$.
4. **Holdout gate**: the knots are written only when the piecewise map beats the affine one on the log-loss of a 2-fold holdout. Otherwise only the affine fit is written.

The result is saved to `data/calibration/gap_d{d}_p{p}.json` and read by `_calibration.get_calibration_map`. Without a file, $s = 0.95,\ o = 0$ is used.

---

## 1. Definitions

### 1.1 Samples

A sample set is identified by $(d, p)$ (`data/sampled_gap/gap_d{d}_p{p}.{csv,npz,mmap}`). It contains $N$ triples

$$
(g_i,\ a_i,\ \hat a_i), \qquad i = 1, \dots, N
$$

| Symbol | Name in the code | Meaning |
|---|---|---|
| $g_i$ | `gaps` | Complementary gap of the inner decoder (real, 6 decimals in the CSV) |
| $a_i \in \{0,1\}$ | `actual` | Actual value of the logical observable |
| $\hat a_i \in \{0,1\}$ | `predicted` | Value of the observable predicted by the inner decoder |
| $y_i = a_i \oplus \hat a_i$ | `mismatch` | 1 when the inner decoder's prediction is wrong |

The gap is produced by `gap_sampler.py`. With $W_0$ the matching weight of the predicted solution and $W_c$ the weight of the solution with the logical class flipped (two membrane detectors flipped),

$$
g = (W_c - W_0) \times \texttt{decibels\_per\_w} .
$$

This script is not involved in how the gap is made; it only uses $(g_i, y_i)$.

### 1.2 The quantity to estimate

> **In one sentence:** the gap is a number that says how confident the inner decoder is. What we want to know is "at that confidence, how often is it actually wrong?". The llr expresses that probability of being wrong on a log scale; the larger it is, the less likely the prediction is wrong. As a guide, llr = 4.6 corresponds to a probability of being wrong of about 1%, and llr = 6.9 to about 0.1%.

Let the probability that the inner decoder's prediction is wrong at gap $g$ be

$$
\pi(g) = P(y = 1 \mid g) .
$$

What the outer decoder needs is the log-likelihood ratio of the event "the prediction is wrong",

$$
\mathrm{llr}(g) = \ln \frac{1 - \pi(g)}{\pi(g)}
$$

($\ln$ is the natural logarithm). Solving for $\pi$,

$$
\pi(g) = \frac{1}{1 + e^{\mathrm{llr}(g)}} = \sigma\bigl(-\mathrm{llr}(g)\bigr), \qquad \sigma(x) = \frac{1}{1 + e^{-x}} .
$$

If the matching weights were perfectly calibrated, $\mathrm{llr}(g) = g$; in practice degeneracy and correlations make it deviate. So the map from gap to llr,

$$
\Lambda : g \mapsto \mathrm{llr} ,
$$

is estimated from samples. It is called the calibration map (`GapCalibration` in the code).

### 1.3 Log-loss

The log-loss is the mean negative log-likelihood of the mismatch labels under a map $\Lambda$:

$$
L(\Lambda) = \frac{1}{N} \sum_{i=1}^{N} \Bigl[\, y_i \ln\bigl(1 + e^{\Lambda(g_i)}\bigr) + (1 - y_i) \ln\bigl(1 + e^{-\Lambda(g_i)}\bigr) \Bigr]
$$

The first term is $-\ln \pi(g_i)$ and the second $-\ln(1 - \pi(g_i))$ (`log_loss_map`). Smaller is better.

---

## 2. Preprocessing: sorting and aggregation

> **In one sentence:** this is reordering and counting to make the later computations fast. It does not change the result of the fits. Sorting by gap is for cutting the gap range into pieces in chapter 4; counting equal gap values together lets chapter 3 work on far fewer rows than the hundreds of millions of samples; and splitting the samples at random into two groups is for the test of chapter 5.

`sort_and_aggregate` builds the following (`SortedSamples`).

**Sorted arrays.** A stable sort by increasing gap (ties in file order):

$$
g_{(1)} \le g_{(2)} \le \dots \le g_{(N)}, \qquad y_{(1)}, \dots, y_{(N)}
$$

**Counts per gap value.** With the distinct gap values $\gamma_1 < \gamma_2 < \dots < \gamma_M$,

$$
n_j = \#\{ i : g_i = \gamma_j \}, \qquad k_j = \#\{ i : g_i = \gamma_j,\ y_i = 1 \} .
$$

In the code these are `gap_values`, `trials`, `events`. Gaps are recorded with 6 decimals, so $M \ll N$. The likelihood of the logistic regression and the log-loss depend only on $(\gamma_j, n_j, k_j)$, so computing on the aggregated table gives exactly the same result:

$$
L(\Lambda) = \frac{1}{N} \sum_{j=1}^{M} \Bigl[\, k_j \ln\bigl(1 + e^{\Lambda(\gamma_j)}\bigr) + (n_j - k_j) \ln\bigl(1 + e^{-\Lambda(\gamma_j)}\bigr) \Bigr]
$$

**Fold mask.** For the holdout, each sample gets an independent random bit

$$
f_i \sim \mathrm{Bernoulli}(1/2)
$$

(`np.random.default_rng(seed).random(N) < 0.5`, `seed = 0`, generated in file order). The counts $n_j^{A}, k_j^{A}$ of the samples with $f_i = 1$ are built at the same time. The other half follows as $n_j^{B} = n_j - n_j^{A}$, $k_j^{B} = k_j - k_j^{A}$. The two folds are not exactly the same size.

Let $K = \sum_j k_j$ be the total number of mismatches. When $K = 0$ the script writes nothing and stops (the defaults are used).

---

## 3. Affine fit (logistic regression)

> **In one sentence:** this draws one straight line "gap → llr". Only two numbers are decided: $s$ is "how much the llr grows when the gap grows by 1", and $o$ is "the llr at gap 0". The $s, o$ that best explain the wrong / right outcomes of all samples are chosen.

### 3.1 Model

$$
\pi_{s,o}(g) = \frac{1}{1 + \exp(s\,g + o)} \quad\Longleftrightarrow\quad \Lambda_{\mathrm{aff}}(g) = s\,g + o
$$

$s$ is `scale` and $o$ is `offset`.

### 3.2 Estimation

The log-likelihood

$$
\ell(s, o) = \sum_{j=1}^{M} \Bigl[\, k_j \ln \pi_{s,o}(\gamma_j) + (n_j - k_j) \ln\bigl(1 - \pi_{s,o}(\gamma_j)\bigr) \Bigr]
$$

is maximized (`fit_logistic_llr`). This is the same as minimizing $L(\Lambda_{\mathrm{aff}})$.

The implementation is Newton's method (IRLS). With the standard-form parameters $w_0 = -o,\ w_1 = -s$,

$$
\eta_j = w_0 + w_1 \gamma_j, \qquad \mu_j = \sigma(\eta_j), \qquad r_j = k_j - n_j \mu_j, \qquad \omega_j = n_j \mu_j (1 - \mu_j)
$$

$$
\nabla = \begin{pmatrix} \sum_j r_j \\ \sum_j \gamma_j r_j \end{pmatrix}, \qquad
H = \begin{pmatrix} \sum_j \omega_j & \sum_j \omega_j \gamma_j \\ \sum_j \omega_j \gamma_j & \sum_j \omega_j \gamma_j^2 \end{pmatrix}, \qquad
w \leftarrow w + H^{-1} \nabla
$$

- The starting point is the default calibration $(w_0, w_1) = (0, -0.95)$. If it does not converge, it restarts from $(0, 0)$.
- It has converged when the largest absolute update is below $10^{-10}$, and returns $(s, o) = (-w_1, -w_0)$. At most 100 iterations.
- It is an error when there are no mismatches, or when every sample is a mismatch.

### 3.3 Recorded values

For the $(s, o)$ fitted on all samples, the script computes

- `fitted_log_loss` $= L(\Lambda_{\mathrm{aff}})$
- `default_log_loss` $= L(g \mapsto 0.95\,g)$

Both are values on the same samples used for the fit (in-sample).

---

## 4. Piecewise fit (a correction to the affine fit)

> **In one sentence:** this checks whether the line of chapter 3 is off in some places, and fixes it. For each range of gaps it compares "the number of predictions that were actually wrong" with "the number the line predicts", and moves the line up or down only where they clearly disagree. It does not refit the line. To see the flow first, start from "In words" and the numerical example in 4.7.

The affine fit is dominated by the low-gap side, where most samples are, and can underestimate the mismatch probability in the high-gap tail. A piecewise-linear map corrects this (`fit_piecewise_llr`).

It runs when `--affine-only` is not given and

$$
K \ge 4m ,
$$

where $m$ is `--min-events` (default 25), so by default at least 100 mismatches are required.

The inputs are the sorted samples $(g_{(i)}, y_{(i)})$ and the $(s, o)$ of the affine fit. Below, the parentheses of the indices are dropped: $g_i, y_i$.

### 4.1 The probability the affine fit predicts

For each sample,

$$
p_i = \frac{1}{1 + \exp(s\,g_i + o)}
$$

### 4.2 Adaptive binning

Bins are cut from the start in sorted order. For the samples of a bin $b$ define

$$
k_b = \sum_{i \in b} y_i \quad \text{(observed mismatches)}, \qquad
E_b = \sum_{i \in b} p_i \quad \text{(mismatches the affine fit expects)}, \qquad
G_b = \sum_{i \in b} g_i\, p_i .
$$

A bin is closed at the first sample, counted from its start, where either

$$
k_b \ge m \quad \text{or} \quad E_b \ge m
$$

holds (that sample included). If neither holds before the end, the bin is closed there. The cut is per sample, so it can fall in the middle of a run of equal gap values.

Why also cut on $E_b$: cutting on the observed count alone would put the sparse high-gap tail in the same bin as the region before it, and the deviation of the tail would be averaged away. Cutting on the expected count as well gives a separate bin for every range in which the affine fit predicts enough events.

**Merging the last bin.** If the last bin has

$$
\max(k_B, E_B) < m / 2 ,
$$

it is added to the previous bin ($k, E, G$ are each summed).

### 4.3 The knot of a bin

A bin (or a merged block, see below) $b$ gives one knot $(g_b^{\ast}, \lambda_b)$.

**Position.** The mean gap weighted by the expected number of events:

$$
g_b^{\ast} = \frac{G_b}{E_b} = \frac{\sum_{i \in b} g_i\, p_i}{\sum_{i \in b} p_i}
$$

**Correction (shift).** The log of the ratio of observed to expected counts:

$$
\delta_b = \ln \frac{k_b + \alpha}{E_b + \alpha}, \qquad \alpha = 0.5 \ \ (\texttt{prior})
$$

$\alpha$ is a pseudo-count that keeps it finite when $k_b = 0$.

**Shrinkage (soft threshold).** The standard error of $\delta_b$ is estimated as $1/\sqrt{k_b + \alpha}$, and $\delta_b$ is shrunk toward 0 by $z$ times that ($z$ is `--shrink-z`, default 1):

$$
\tilde\delta_b = \mathrm{sign}(\delta_b)\, \max\!\left(0,\ \lvert \delta_b \rvert - \frac{z}{\sqrt{k_b + \alpha}}\right)
$$

When the deviation is within the statistical error, $\tilde\delta_b = 0$ and the bin agrees exactly with the affine fit.

**The llr of the knot.**

$$
\lambda_b = s\, g_b^{\ast} + o - \tilde\delta_b
$$

**Meaning.** For $\pi \ll 1$, $\mathrm{llr} \approx -\ln \pi$, so subtracting $\tilde\delta_b$ from the llr is almost the same as multiplying the mismatch probability by $e^{\tilde\delta_b}$. It is the correction "multiply the probability predicted by the affine fit by the ratio observed / expected". When more mismatches are observed than expected, $\delta_b > 0$ and the llr goes down (less confidence). On the low-gap side, where $\pi$ is not small, this correspondence is approximate.

$k_b / E_b$ is used, and not the empirical mismatch rate at the mean gap of the bin, so that the estimate of the ratio is not biased when $\pi(g)$ changes by orders of magnitude inside a bin.

**Numerical example (made-up values).** With $k_b = 25,\ E_b = 12.5,\ z = 1$,

$$
\delta_b = \ln\frac{25.5}{13.0} = 0.674, \qquad \frac{z}{\sqrt{25.5}} = 0.198, \qquad \tilde\delta_b = 0.476 .
$$

The llr is 0.476 lower than the affine one (the mismatch probability is estimated about $e^{0.476} = 1.61$ times larger).

### 4.4 Making it monotone (PAVA)

The llr should increase with the gap, so whenever two neighboring blocks have

$$
\lambda_{b-1} \ge \lambda_b ,
$$

they are merged. Merging adds $(k, E, G)$, and the knot $(g^{\ast}, \lambda)$ is recomputed from the merged totals with the formulas of 4.3 (the shrinkage too). The scan goes left to right and is repeated until no merge happens. The knot llrs end up strictly increasing.

### 4.5 The first knot

Let $g_0 = \min(0,\ g_{(1)})$. If the knot of the first block has $g_1^{\ast} > g_0$, the knot

$$
\bigl(g_0,\ \ s\,g_0 + o - \tilde\delta_1\bigr)
$$

is added in front. On $[g_0, g_1^{\ast}]$ the map is then the affine line shifted by $\tilde\delta_1$.

### 4.6 The resulting map

With knots $(x_0, \lambda_0), \dots, (x_R, \lambda_R)$ ($x_0 < \dots < x_R$), `GapCalibration.__call__` returns

$$
\Lambda_{\mathrm{pw}}(g) =
\begin{cases}
\lambda_0 & g \le x_0 \\[4pt]
\lambda_r + \dfrac{\lambda_{r+1} - \lambda_r}{x_{r+1} - x_r}\,(g - x_r) & x_r \le g \le x_{r+1} \\[10pt]
\lambda_R + s_{\mathrm{tail}}\,(g - x_R) & g > x_R
\end{cases}
$$

$s_{\mathrm{tail}}$ is `tail_scale`; the fit sets it to the affine $s$. Beyond the last knot the map is therefore parallel to the affine line, and the correction of the last block, $\tilde\delta_R$, applies to the whole tail.

### 4.7 The linear interpolation as a whole

First the flow without formulas, then a small numerical example, and finally the formulas.

#### In words

1. **The starting point is one straight line.** The affine fit of chapter 3 has already drawn one line "gap in, llr out". It was determined from all samples and is not refitted afterwards.

2. **Check where the line fits.** Cut the samples by increasing gap (these are the bins), and in each piece compare two numbers:
   - the number of predictions that were actually wrong, $k$
   - the number that would be wrong if the line were right, $E$

3. **From the comparison decide one number: how much to move the line up or down.** This is the correction $\tilde\delta$.
   - $k$ larger than $E$ → the line is overconfident → lower the llr there ($\tilde\delta > 0$)
   - $k$ smaller than $E$ → the line is too cautious → raise the llr ($\tilde\delta < 0$)
   - the difference is within the statistical error → do nothing ($\tilde\delta = 0$)

4. **Place one point per piece.** This is a knot.
   - Horizontal position: a gap that represents the piece (the average gap inside it)
   - Vertical position: the value on the line, lowered by $\tilde\delta$

5. **Join the points from left to right with straight segments.** This is the linear interpolation, and the result is a broken line. For a gap between two points, find what fraction of the way from the left point to the right point it is, and advance the height by the same fraction. No data is used here; the points already placed are just joined.

6. **Extend both ends in parallel.**
   - Left of the leftmost point: keep the correction of the first piece and extend parallel to the line down to gap 0.
   - Right of the rightmost point: keep the correction of the last piece and extend parallel to the line forever.

In short, the line is not refitted inside each piece. The overall line is moved slightly up or down from place to place, and the amount of movement is joined smoothly between neighbors.

#### Numerical example (made-up values)

To keep the arithmetic simple, the line is $s = 1,\ o = 0$ (llr = gap). There are three pieces with the following corrections.

| Piece | Representative gap | Correction $\tilde\delta$ | Value on the line | Height of the point |
|---|---:|---:|---:|---:|
| 1 | 2 | 0 | 2.0 | 2.0 |
| 2 | 6 | +0.4 | 6.0 | 5.6 |
| 3 | 10 | +0.2 | 10.0 | 9.8 |

The correction of the first piece is 0, so the first point is $(0,\ 0)$. There are 4 points in total: $(0, 0),\ (2, 2.0),\ (6, 5.6),\ (10, 9.8)$.

The llr for a few gaps:

| gap | Between which points | Computation | llr of the broken line | llr of the line | Difference (correction) |
|---:|---|---|---:|---:|---:|
| 1 | $(0, 0)$ and $(2, 2.0)$ | halfway: $0 + 2.0 \times \tfrac{1}{2}$ | 1.0 | 1.0 | 0 |
| 4 | $(2, 2.0)$ and $(6, 5.6)$ | halfway: $2.0 + 3.6 \times \tfrac{2}{4}$ | 3.8 | 4.0 | 0.2 |
| 7 | $(6, 5.6)$ and $(10, 9.8)$ | a quarter: $5.6 + 4.2 \times \tfrac{1}{4}$ | 6.65 | 7.0 | 0.35 |
| 12 | right of the last point | parallel extension: $9.8 + 1 \times (12 - 10)$ | 11.8 | 12.0 | 0.2 |

Plotting only the rightmost column (the correction) against the gap gives:

```
D(g)
0.4 |           *
0.3 |         .     .
0.2 |       .           *------>
0.1 |     .
0.0 *---*
    +---+-------+-------+------  gap
    0   2       6       10
```

The correction itself is a broken line through the values $0,\ 0.4,\ 0.2$ of the pieces. The correction 0.2 at gap 4 is halfway between 0 and 0.4, and 0.35 at gap 7 is a quarter of the way from 0.4 to 0.2. Right of the last point it stays at 0.2.

What a correction of 0.4 means: an llr lower by 0.4 means the odds of being wrong are re-estimated $e^{0.4} \approx 1.5$ times larger.

#### In formulas

(a)–(e) below put the above into formulas; (f) gives the values of the actual files.

#### (a) The sequence of knots

Number the blocks after PAVA by increasing gap, $b = 1, \dots, R$. Each block gives one knot, and the knot of 4.5 is added in front.

| $r$ | Position $x_r$ | Height $\lambda_r$ | Correction |
|---|---|---|---|
| $0$ | $g_0 = \min(0,\ g_{(1)})$ | $s\,x_0 + o - \tilde\delta_1$ | $\tilde\delta_0 := \tilde\delta_1$ |
| $1, \dots, R$ | $g_r^{\ast} = G_r / E_r$ | $s\,x_r + o - \tilde\delta_r$ | $\tilde\delta_r$ |

There are $R + 1$ knots. Every knot is "the point of the overall affine line, lowered by $\tilde\delta_r$". No line is fitted inside a bin; a bin only decides the two numbers $x_r$ and $\tilde\delta_r$.

- $x_r$ is not a bin boundary but the weighted mean gap inside the bin.
- The blocks are non-overlapping ranges in gap order, so $x_1 \le \dots \le x_R$ (strictly increasing in the current files).
- By PAVA, $\lambda_1 < \dots < \lambda_R$.

#### (b) Rewritten as an interpolation of the correction

Define the linear interpolation of the knot corrections $\tilde\delta_r$:

$$
D(g) =
\begin{cases}
\tilde\delta_r + \dfrac{\tilde\delta_{r+1} - \tilde\delta_r}{x_{r+1} - x_r}\,(g - x_r) & x_r \le g \le x_{r+1} \\[10pt]
\tilde\delta_R & g > x_R
\end{cases}
$$

The affine line $s\,g + o$ is itself linear, so interpolating the knot heights linearly is the same as interpolating the correction linearly and subtracting it from the affine line. Hence for $g \ge x_0$,

$$
\Lambda_{\mathrm{pw}}(g) = s\,g + o - D(g)
$$

(when $s_{\mathrm{tail}} = s$). The piecewise map is "the overall affine line" minus "the broken-line correction $D(g)$".

In terms of probabilities, for the odds $\pi / (1 - \pi) = e^{-\mathrm{llr}}$, exactly

$$
\frac{\pi_{\mathrm{pw}}(g)}{1 - \pi_{\mathrm{pw}}(g)} = e^{D(g)} \cdot \frac{\pi_{\mathrm{aff}}(g)}{1 - \pi_{\mathrm{aff}}(g)} .
$$

Where $D(g) > 0$ the map estimates more mismatches than the affine fit, and where $D(g) = 0$ it agrees with it.

#### (c) The shape on each interval

| Interval | $\Lambda_{\mathrm{pw}}(g)$ | Relation to the affine fit |
|---|---|---|
| $g < x_0$ | $\lambda_0$ (constant) | Value fixed outside the interpolation. $x_0 \le g_{(1)}$, so this range does not occur in the samples used for the fit |
| $x_0 \le g \le x_1$ | $s\,g + o - \tilde\delta_1$ | A parallel line lowered by $\tilde\delta_1$ |
| $x_r \le g \le x_{r+1}$ $(1 \le r < R)$ | The segment joining $\lambda_r$ and $\lambda_{r+1}$ | The correction moves linearly from $\tilde\delta_r$ to $\tilde\delta_{r+1}$ |
| $g > x_R$ | $s\,g + o - \tilde\delta_R$ | A parallel line lowered by $\tilde\delta_R$ |

#### (d) The slope of a segment

The slope on $[x_r, x_{r+1}]$ is

$$
\frac{\lambda_{r+1} - \lambda_r}{x_{r+1} - x_r} = s - \frac{\tilde\delta_{r+1} - \tilde\delta_r}{x_{r+1} - x_r} .
$$

- If the corrections at both ends are equal (including both 0), the slope is $s$ and the segment is parallel to the affine line.
- Where the correction increases the slope is below $s$; where it decreases the slope is above $s$.
- PAVA guarantees $\lambda_{r+1} > \lambda_r$, so the slope is always positive.
- The slope is not fitted to the data; it is determined only by the difference of the two knots. When knots are close together, a small difference in correction makes the local slope deviate a lot from $s$.

#### (e) How it is evaluated

`GapCalibration.__call__` returns the value in two steps.

1. `np.interp(g, knots_gap, knots_llr)`: linear interpolation for $x_0 \le g \le x_R$, and the end value ($\lambda_0$ or $\lambda_R$) outside.
2. The elements with $g > x_R$ are overwritten by $\lambda_R + s_{\mathrm{tail}}\,(g - x_R)$.

Arrays can be passed directly, so the simulation converts all sampled gaps to llrs at once.

#### (f) The actual values of the current knots

The result of computing $\tilde\delta_r = s\,x_r + o - \lambda_r$ back from the knots of `data/calibration/gap_d{5,7,11}_p0.001.json`.

| d | Knots | Knots with nonzero correction | Range of $\tilde\delta_r$ | Last knot $x_R$ | Tail correction $\tilde\delta_R$ | Range of segment slopes | $s$ |
|---|---:|---:|---|---:|---:|---|---:|
| 5 | 195 | 55 | −0.099 to +0.116 | 9.80 | 0 | 0.012 to 8.75 | 1.0131 |
| 7 | 253 | 47 | −0.112 to +0.316 | 11.64 | +0.141 | 0.036 to 5.77 | 1.0076 |
| 11 | 335 | 46 | −0.244 to +0.157 | 13.29 | 0 | 0.0038 to 5.18 | 0.9965 |

- For most knots the shrinkage sets the correction to 0, and near them the map agrees with the affine fit.
- d=5 and d=11 have $\tilde\delta_R = 0$, so beyond the last knot the map is the affine fit itself.
- d=7 has $\tilde\delta_R = +0.141$: for all $g > 11.64$ the llr is lower by 0.141 (the odds are estimated $e^{0.141} = 1.15$ times larger).
- Where the correction differs from that of the neighboring knot, the slope of the segment is far from $s \approx 1$ ("Range of segment slopes" in the table).

The last 8 knots of d=11 as an example:

| $r$ | $x_r$ | $\lambda_r$ | $\tilde\delta_r$ |
|---:|---:|---:|---:|
| 327 | 7.7781 | 8.0073 | −0.2438 |
| 328 | 8.1335 | 8.1177 | 0 |
| 329 | 8.5234 | 8.4394 | +0.0668 |
| 330 | 8.8318 | 8.7749 | +0.0386 |
| 331 | 9.2769 | 9.2571 | 0 |
| 332 | 9.9631 | 9.9409 | 0 |
| 333 | 10.9794 | 10.9537 | 0 |
| 334 | 13.2873 | 13.2536 | 0 |

On the segment $r = 327 \to 328$ the correction returns from −0.244 to 0, so the slope is $(8.1177 - 8.0073) / (8.1335 - 7.7781) = 0.31$. From $r = 331$ on the correction stays 0, so the slope is $s$.

---

## 5. Holdout gate

> **In one sentence:** this is a test of whether the correction of chapter 4 really helps or only follows the fluctuations of that particular data. The samples are split into two groups; the map built on one is scored on the other. This is done twice with the roles swapped, and the correction is adopted only when the score with it is better than without it.

The piecewise map has many degrees of freedom, so its log-loss on the samples used for the fit tends to be lower than the affine one. A 2-fold holdout makes the comparison fair (`_fit_fold`).

Let the folds be $A = \{ i : f_i = 1 \}$ and $B = \{ i : f_i = 0 \}$. For each of (train, test) $= (A, B)$ and $(B, A)$:

1. Fit the affine $(s^{\mathrm{tr}}, o^{\mathrm{tr}})$ on train only.
2. Fit the knots on train only, on top of that affine fit ($s_{\mathrm{tail}} = s^{\mathrm{tr}}$).
3. Compute both log-losses on test (the mean over the number of test samples).

The results of the two folds are averaged with equal weight:

$$
\bar L_{\mathrm{aff}} = \tfrac{1}{2}\bigl[ L_B(\Lambda_{\mathrm{aff}}^{A}) + L_A(\Lambda_{\mathrm{aff}}^{B}) \bigr], \qquad
\bar L_{\mathrm{pw}} = \tfrac{1}{2}\bigl[ L_B(\Lambda_{\mathrm{pw}}^{A}) + L_A(\Lambda_{\mathrm{pw}}^{B}) \bigr]
$$

The superscript is the fold used for the fit and the subscript the fold used for the evaluation. The decision is a plain comparison:

$$
\bar L_{\mathrm{pw}} < \bar L_{\mathrm{aff}} \ \Rightarrow\ \text{adopt the piecewise map}
$$

- Adopted: the knots fitted on **all samples** are written to the JSON (they are computed ahead, in parallel with the holdout).
- Not adopted: the JSON holds the affine fit only, and the behavior is that of the affine fit.

There is no test of whether the difference is significant.

---

## 6. The output JSON

`data/calibration/gap_d{d}_p{p}.json`

| Key | Content | Always written |
|---|---|---|
| `d`, `p` | Identify the sample set | yes |
| `scale`, `offset` | The affine fit $s, o$ on all samples | yes |
| `num_samples`, `num_mismatch` | $N$, $K$ | yes |
| `fitted_log_loss`, `default_log_loss` | The in-sample log-losses of 3.3 | yes |
| `holdout_affine_log_loss`, `holdout_piecewise_log_loss` | $\bar L_{\mathrm{aff}}, \bar L_{\mathrm{pw}}$ | when the piecewise fit was tried |
| `knots_gap`, `knots_llr` | The knots $x_r$, $\lambda_r$ | only when the piecewise map is adopted |
| `tail_scale` | $s_{\mathrm{tail}}$ ($= s$) | same |
| `min_events`, `shrink_z` | $m$, $z$ | same |
| `piecewise_log_loss` | $L(\Lambda_{\mathrm{pw}})$ (in-sample) | same |

`get_calibration_map` returns a piecewise `GapCalibration` when `knots_gap` is present and an affine one otherwise. Files fitted in the earlier project also carry a `"kind": "space"` entry, which is ignored.

### Content of the current files (as of 2026-09-30)

| d | $N$ | $K$ | scale | offset | $\bar L_{\mathrm{aff}}$ | $\bar L_{\mathrm{pw}}$ | Adopted | Knots |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 5 | 1,738,918 | 12,050 | 1.013136 | +0.007283 | 2.033864e-2 | 2.033251e-2 | piecewise | 195 |
| 6 | 3,775,000 | 11,161 | 1.018665 | −0.026459 | 8.532689e-3 | 8.533323e-3 | affine | – |
| 7 | 8,741,558 | 10,064 | 1.007619 | +0.010365 | 3.444397e-3 | 3.444382e-3 | piecewise | 253 |
| 8 | 22,306,000 | 10,260 | 1.008234 | −0.000832 | 1.374929e-3 | 1.375105e-3 | affine | – |
| 9 | 55,346,000 | 10,106 | 1.003417 | −0.034457 | 5.476610e-4 | 5.476953e-4 | affine | – |
| 10 | 208,422,000 | 14,250 | 1.005733 | −0.014711 | 2.063601e-4 | 2.063646e-4 | affine | – |
| 11 | 468,038,000 | 12,045 | 0.996534 | +0.012336 | 7.859877e-5 | 7.859382e-5 | piecewise | 335 |

The holdout differences are, relatively, between about $4 \times 10^{-6}$ (d=7) and $3 \times 10^{-4}$ (d=5).

---

## 7. The report table (display only)

The table printed at run time splits the samples into 12 bins at gap quantiles and lists, per bin,

- `empirical`: the mismatch rate in the bin, $\sum y_i / n$
- `affine` / `piecewise`: the predicted probability $1 / (1 + e^{\Lambda(\bar g)})$ at the mean gap $\bar g$ of the bin

(`binned_stats`, `format_binned_table`). It is not used by the fits. The prediction is evaluated at the single point of the mean gap, so it does not match `empirical` directly when the probability changes a lot inside a bin.

---

## 8. How it is used downstream

- `src/gap_simulator/util/_patch.py` and `src/gap_simulator/util/_simulator.py` get `get_calibration_map(d, p)` and apply it to the sampled gaps to obtain llrs.
- Idling over several ticks, or a merged measurement over several cells, is an XOR of independent error events, so the llrs are combined with box-plus (`combine_llrs`):

$$
\tanh\frac{L}{2} = \prod_i \tanh\frac{\ell_i}{2}
$$

- `num_samples` is kept as `GapCalibration.num_samples`.

---

## 9. Parameters

| Symbol | Option / constant | Default | Role |
|---|---|---|---|
| $m$ | `--min-events` | 25 | Number of events that closes a bin. The piecewise fit needs $K \ge 4m$ |
| $z$ | `--shrink-z` | 1.0 | Multiple of the standard error by which the shift is shrunk |
| $\alpha$ | `prior` | 0.5 | Pseudo-count of the shift |
| – | `seed` | 0 | Random seed of the fold mask |
| – | `--affine-only` | off | Skip the piecewise fit and the holdout |
| – | `DEFAULT_SCALE`, `DEFAULT_OFFSET` | 0.95, 0.0 | The map used when there is no JSON. Also the starting point of Newton's method |
| – | `--jobs`, `--set-jobs` | all cores, 1 | Degree of parallelism. The results are bit-identical to a serial run |
