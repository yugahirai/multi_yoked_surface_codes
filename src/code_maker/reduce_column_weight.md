# How the weight reduction in reduce_column_weight.py works

Script: [reduce_column_weight.py](reduce_column_weight.py)

For a hierarchical check matrix $H=[H_1;H_2;H_3]$, the script minimizes the total number of 1s (= the sum of the column weights) using **only row operations that do not change the information the decoder receives**, and then lowers the maximum column weight. It has three stages.

| Stage | Goal | Guarantee | Function |
|---|---|---|---|
| 1. Exact solve | Minimize the total weight | Global optimum (proved) | [exact_min_rows](reduce_column_weight.py#L85) |
| 2. Flattening | Flatten the column-weight distribution at the same total weight | Local optimum | [flatten_column_max](reduce_column_weight.py#L199) |
| 3. Column-cap annealing | Reach maximum column weight $\le t$ (the total weight may grow a little) | Heuristic | [cap_column_weight_section](reduce_column_weight.py#L344) |

When L3 has too many rows, a greedy local search ([reduce_levels](reduce_column_weight.py#L147)) replaces stage 1.

---

## 1. Problem setting

Write the Z checks (the X checks are handled the same way) as a matrix over GF(2):

$$
H=\begin{pmatrix}H_1\\H_2\\H_3\end{pmatrix},\qquad H_\ell\in\mathbb F_2^{\,n_\ell\times n}
$$

The symbols mean the following.

| Symbol | Meaning | In the script |
|---|---|---|
| $n$ | Number of columns (= length of one row) | `n_cols` |
| $H_\ell$ | Matrix of the level-$\ell$ checks ($\ell=1,2,3$) | `l1`, `l2`, `l3` |
| $n_1$ | Number of L1 rows (rows of $H_1$) | `n_l1` (detected by `levels`) |
| $n_2$ | Number of L2 rows (rows of $H_2$) | `n_l2` (detected by `levels`) |
| $n_3$ | Number of L3 rows (rows of $H_3$) | all remaining rows (total $-\,n_1-n_2$) |
| $h^{(\ell)}_i$ | Row $i$ of $H_\ell$ (a vector in $\mathbb F_2^n$) | `l1[i]` etc. |

In the file, the Z-check (X-check) rows are listed top to bottom as $n_1$ L1 rows, $n_2$ L2 rows, and the rest as L3. The level sizes are not configured by hand: [levels](reduce_column_weight.py#L26) reads them off the structure of our construction (the L1 rows have pairwise disjoint supports, and the L3 rows have the form $[M, M]$), so the input must be a file written by `l3_code_maker.py`. Below we assume that the rows of each level are linearly independent modulo the lower levels (no nonzero combination of L2 rows is a sum of L1 rows, and so on). Section 3.4 uses this assumption.

$H_1$ is the most frequently measured level, and whenever a higher level is measured in a block, the lower levels are measured too.

The quantity to minimize is the total number of 1s

$$
W(H)=\sum_{i}\mathrm{wt}(h_i)=\sum_{c=1}^{n} w_c ,\qquad w_c=\text{weight of column } c .
$$

An error on column $c$ flips $w_c$ checks, so $W/n$ is the average number of detectors per error in the DEM, which drives the search cost of the decoder.

Define the row space of each level as

$$
V_1=\mathrm{rowspan}(H_1),\quad V_2=\mathrm{rowspan}(H_1,H_2),\quad V_3=\mathrm{rowspan}(H_1,H_2,H_3)
$$

($V_1\subset V_2\subset V_3$).

## 2. Allowed transformations

The allowed transformations are $H'=TH$ with an invertible block-lower-triangular matrix $T$:

$$
T=\begin{pmatrix}I&0&0\\A_{21}&A_{22}&0\\A_{31}&A_{32}&A_{33}\end{pmatrix},\qquad A_{22},A_{33}\ \text{invertible}
$$

In other words, only

- invertible row recombinations within one level ($A_{22},A_{33}$), and
- additions of lower-level rows to higher-level rows ($A_{21},A_{31},A_{32}$)

are used.

**Why the information does not change.** In a block that measures up to level $\ell$, the syndrome obtained is $s=H_{\le\ell}\,e$. After the transformation it is $s'=T_{\le\ell}\,s$, and $T_{\le\ell}$ (the upper-left block of $T$) is block lower triangular and therefore invertible. So in every measurement block $s\leftrightarrow s'$ is a bijection and the decoder receives the same information. This is equivalent to preserving the flag $V_1\subset V_2\subset V_3$. The stabilizer group $V_3$ itself is unchanged, so the code and its logical operators are unchanged.

Because the information is the same, the logical error rate of a maximum-likelihood decoder is exactly unchanged. An approximate decoder could in principle predict differently, so that has to be checked separately by measurement.

**CSS condition.** Z rows are only mixed with Z rows and X rows with X rows, so $H_Z'H_X'^{\top}=T_ZH_ZH_X^{\top}T_X^{\top}=0$ is preserved.

**Why $H_1$ is not touched.** The L1 rows have pairwise disjoint supports. A nonzero element of $V_1$ is a union of blocks, so any recombination only increases the weight, and the current $H_1$ is already minimal. In addition, the simulator's L1 presolve relies on the structure "every column flips exactly one L1 detector".

## 3. Decomposition: a minimum-weight basis of a quotient space

Consider L2 only at first. The transformed L2 block is $H_2'=A_{21}H_1+A_{22}H_2$. We unpack it step by step.

### 3.1 What a coset is

For a vector $t\in\mathbb F_2^n$, the set of all vectors obtained by adding an element of $V_1$ (= a sum of L1 rows) to $t$,

$$
t+V_1=\{\,t+u\ :\ u\in V_1\,\},
$$

is called the **coset** of $t$. It is the collection of vectors that are "the same if we ignore adding or removing L1 rows". L1 is always measured together with the higher levels, so two vectors of the same coset give the same information as checks. Each vector in a coset is called a **representative**.

A small example ($n=4$): with L1 rows $1100$ and $0011$, $V_1=\{0000,\,1100,\,0011,\,1111\}$. The coset of $t=1110$ is

$$
t+V_1=\{\,1110,\ 0010,\ 1101,\ 0001\,\},
$$

four vectors of weight 3, 1, 3, 1. Measuring $1110$ or $0010$ as an L2 check gives the same information, so we can use the light $0010$ (or $0001$). That is the basic idea of the script.

### 3.2 Writing row $i$ by components

Writing row $i$ of $H_2'$, $h_i'$, by the definition of the matrix product gives

$$
h_i'=\underbrace{\sum_{j=1}^{n_1}(A_{21})_{ij}\,h^{(1)}_j}_{u_i}\ +\ \underbrace{\sum_{k=1}^{n_2}(A_{22})_{ik}\,h^{(2)}_k}_{t_i}
$$

($h^{(1)}_j$ is row $j$ of L1, and $h^{(2)}_k$ is row $k$ of the original L2).

- $u_i$ is a **sum of L1 rows**, so it is an element of $V_1$.
- $t_i$ is a **sum of original L2 rows**: row $i$ of the matrix $A_{22}H_2$.

Hence $h_i'=t_i+u_i$ with $u_i\in V_1$, that is, $h_i'$ is an element of the coset $t_i+V_1$. This is what "row $i$ is an element of the coset $(A_{22}H_2)_i+V_1$" means.

The roles of $A_{22}$ and $A_{21}$ separate cleanly.

| Matrix | What it decides |
|---|---|
| $A_{22}$ | **Which coset** each row belongs to ($t_i$) |
| $A_{21}$ | **Which representative** within that coset ($u_i$) |

### 3.3 $A_{21}$ is free → any representative can be chosen

There is no constraint on $A_{21}$. Row $i$ of $A_{21}$ only affects $h_i'$, and as that row ranges over all of $\mathbb F_2^{n_1}$, $u_i$ ranges over all of $V_1$. Therefore

- $h_i'$ can be **any representative** of the coset $t_i+V_1$,
- and the choice is independent for every row,

so the best choice is the lightest representative of each row. In the example above this is choosing $0010$ instead of $1110$.

### 3.4 $A_{22}$ is invertible → the cosets form a basis of the quotient space

Section 3.3 showed that once the coset is fixed, we should pick its lightest representative. The remaining question is which sets of cosets can be produced by varying $A_{22}$. To state it we introduce a framework that treats a coset as one object (the quotient space).

#### (a) $V_2$ is partitioned into cosets

For each element $t$ of $V_2$ (the vectors that are sums of L1 and L2 rows), consider the coset $t+V_1$. Two cosets coincide exactly when

$$
t+V_1=t'+V_1\iff t+t'\in V_1 ,
$$

and otherwise they have no element in common. So $V_2$ is partitioned exactly into non-overlapping cosets.

- Number of elements of $V_2$: $2^{n_1+n_2}$
- Number of elements of one coset: $\lvert V_1\rvert=2^{n_1}$
- Number of cosets: $2^{n_1+n_2}/2^{n_1}=2^{n_2}$

The **set of all these cosets** is written $V_2/V_1$ ("$V_2$ divided by $V_1$"). One element of $V_2/V_1$ is not a single vector but a bundle of $2^{n_1}$ vectors (one coset). Below, the coset containing a vector $t$ is written $[t]=t+V_1$.

**Example.** As in 3.1, take $n=4$, L1 rows $1100,\ 0011$, and two L2 rows $a=1010,\ b=1110$ ($n_1=n_2=2$). $V_2$ has 16 vectors and is partitioned into the following 4 cosets.

| Coset | Contents (4 vectors each) | Weight of the lightest representative |
|---|---|---|
| $[0]=V_1$ | $0000,\ 1100,\ 0011,\ 1111$ | 0 |
| $[a]$ | $1010,\ 0110,\ 1001,\ 0101$ | 2 |
| $[b]$ | $1110,\ 0010,\ 1101,\ 0001$ | 1 |
| $[a+b]$ | $0100,\ 1000,\ 0111,\ 1011$ | 1 |

$V_2/V_1$ is the set of these 4 cosets.

#### (b) Adding cosets

Define the sum of two cosets as "take one representative from each, add them, and return the coset that contains the result":

$$
[t]+[t']=[t+t']
$$

The answer does not depend on the representatives. With other representatives $t+u,\ t'+u'$ ($u,u'\in V_1$) the sum is $(t+t')+(u+u')$, and $u+u'\in V_1$, so it lies in the same coset.

In the example $[a]+[b]=[a+b]$. With representatives $1010+1110=0100$ or with others $0110+0001=0111$, the result lies in the same coset $[a+b]$.

With this addition $V_2/V_1$ is a vector space over $\mathbb F_2$, called the **quotient space**.

- Its zero element is $[0]=V_1$ itself (the bundle of vectors that are sums of L1 rows only).
- It has $2^{n_2}$ elements, so its dimension is $n_2$.

Intuitively, it is "the space that keeps only the L2 degrees of freedom once differences by L1 rows are ignored".

#### (c) The cosets of the original L2 rows form a basis

Any element of $V_2$ can be written $v=\sum_k c_k h^{(2)}_k+u$ ($c_k\in\{0,1\}$, $u\in V_1$), so its coset is

$$
[v]=\sum_{k=1}^{n_2}c_k\,[h^{(2)}_k] .
$$

That is, every coset is a combination of the cosets $[h^{(2)}_1],\dots,[h^{(2)}_{n_2}]$ of the original L2 rows. By the assumption of section 1 (the L2 rows are linearly independent modulo L1) the expression is unique, so they form a basis of $V_2/V_1$.

Each coset therefore corresponds one-to-one to a coefficient vector $c=(c_1,\dots,c_{n_2})\in\mathbb F_2^{n_2}$. $c$ is the coordinate that says which L2 rows were added together. In the example,

$$
[0]\leftrightarrow(0,0),\quad [a]\leftrightarrow(1,0),\quad [b]\leftrightarrow(0,1),\quad [a+b]\leftrightarrow(1,1) .
$$

#### (d) Each row of $A_{22}$ is the coordinate of a coset

From $t_i=\sum_k(A_{22})_{ik}h^{(2)}_k$ in 3.2, the coset of the new row $i$ is

$$
[t_i]=\sum_{k=1}^{n_2}(A_{22})_{ik}\,[h^{(2)}_k] .
$$

Comparing with (c), **row $i$ of $A_{22}$ is exactly the coordinate of the coset of the new row $i$**.

$n_2$ vectors of $\mathbb F_2^{n_2}$ form a basis if and only if the matrix with those rows is invertible. Hence

$$
A_{22}\ \text{invertible}\iff [t_1],\dots,[t_{n_2}]\ \text{is a basis of}\ V_2/V_1 .
$$

Check with the example.

- $A_{22}=\begin{pmatrix}0&1\\1&1\end{pmatrix}$ (invertible): the cosets of the new rows are $[b],\ [a+b]$. This is a basis, and the original $[a]$ is recovered as $[b]+[a+b]$.
- $A_{22}=\begin{pmatrix}1&1\\1&1\end{pmatrix}$ (not invertible): both rows fall into $[a+b]$, and the information of $[a]$ or $[b]$ alone is lost. This is not allowed.

Invertibility is required for the reason given in section 2: the information obtained when measuring up to L2 must not decrease.

### 3.5 Summary

Combining 3.3 and 3.4, the set of all allowed $H_2'$ can be restated as

> Choose a basis $\{q_1,\dots,q_{n_2}\}$ of $V_2/V_1$, and take one representative from each coset $q_i$.

Conversely, anything chosen this way is realized by some $(A_{21},A_{22})$. To minimize the weight we take the lightest representative of each coset, so the only remaining freedom is "which basis to choose".

Comparing on the example of 3.4:

| Choice | The 2 rows of L2 | Sum of weights |
|---|---|---|
| Unchanged | $1010,\ 1110$ | $2+3=5$ |
| Same basis $[a],[b]$, lightest representatives only | $1010,\ 0010$ | $2+1=3$ |
| Basis also replaced by $[b],[a+b]$ | $0010,\ 0100$ | $1+1=2$ |

Re-choosing the representatives ($A_{21}$) alone stops at 3; only by also replacing the basis ($A_{22}$) is the minimum 2 reached. Which basis is lightest is decided by the greedy method of section 5.

L3 has exactly the same form. Row $i$ of $H_3'=A_{31}H_1+A_{32}H_2+A_{33}H_3$ is

$$
h_i'=\underbrace{(A_{31}H_1+A_{32}H_2)_i}_{\in V_2}+(A_{33}H_3)_i ,
$$

so reading "$V_1$" as "$V_2$", "$A_{21}$" as "$(A_{31},A_{32})$" and "$A_{22}$" as "$A_{33}$" gives the choice of a basis of $V_3/V_2$ and of representatives. This condition depends only on the space $V_2$, not on the particular choice of $H_2'$, so the two levels can be solved independently.

### 3.6 The minimization problem

Define the weight of a coset $q$ as the weight of its lightest representative,

$$
w(q)=\min_{v\in q}\mathrm{wt}(v) .
$$

Then

$$
\min_T W(TH)=W(H_1)+\underbrace{\min_{\{q_i\}:\ \text{basis of } V_2/V_1}\sum_i w(q_i)}_{\text{L2}}+\underbrace{\min_{\{q_i\}:\ \text{basis of } V_3/V_2}\sum_i w(q_i)}_{\text{L3}} .
$$

So the problem reduces to "**find a minimum-weight basis of a quotient space**". Two things are needed: (a) computing the coset weights $w(q)$ and (b) choosing a minimum-weight basis.

## 4. Computing the coset weights

### 4.1 The L1 fold (closed form)

Let $B_j$ be the support of the L1 row $h^{(1)}_j$ (pairwise disjoint), and $U$ the set of columns not covered by any L1 row. Adding a subset $S$ of the L1 rows to a vector $v$ turns the number of 1s inside block $B_j$, $a_j=\mathrm{wt}(v|_{B_j})$, into $\lvert B_j\rvert-a_j$ if $j\in S$, and does not affect the other blocks. The optimization therefore separates per block:

$$
w_{V_1}(v)=\min_{u\in V_1}\mathrm{wt}(v+u)=\mathrm{wt}(v|_U)+\sum_{j=1}^{n_1}\min\bigl(a_j,\ \lvert B_j\rvert-a_j\bigr) .
$$

The optimal choice is "**add an L1 row only when the vector covers more than half of its block**" ($2a_j>\lvert B_j\rvert$). This is [fold](reduce_column_weight.py#L57), and it replaces a search over $2^{n_1}$ possibilities by an $O(n_1)$ computation.

Example: a row that covers 12 columns of a block of size 16 covers only 4 after that L1 row is added.

### 4.2 L2 cosets ($V_2/V_1$)

The only lower level is $V_1$, so $w(v+V_1)=w_{V_1}(v)$: a single fold.

### 4.3 L3 cosets ($V_3/V_2$)

An element of $V_2$ is "a sum of a subset of L2 rows + an element of $V_1$", so the L2 side is enumerated in full and the L1 side is closed by the fold:

$$
w(v+V_2)=\min_{S\subseteq\{1,\dots,n_2\}}\ w_{V_1}\Bigl(v+\sum_{i\in S}h^{(2)}_i\Bigr) .
$$

It is the minimum over $2^{n_2}$ folds (256 for $n_2=8$).

### 4.4 Enumerating all cosets

$V_3/V_2\cong\mathbb F_2^{n_3}$ has $2^{n_3}-1$ nonzero cosets, and the coset with index $x\in\mathbb F_2^{n_3}$ has the representative $\sum_i x_i h^{(3)}_i$. The script enumerates **all** of them and builds a table of weights.

$$
\text{cost}\ \sim\ 2^{n_3}\times 2^{n_2}\times(\text{one fold})
$$

The implementation packs the rows into 16-bit words, processes them in bulk with a popcount lookup table and numpy vector operations, and runs in parallel processes ([_exact_weight_chunk](reduce_column_weight.py#L65)). The cost is proportional to $2^{n_3}$, so above `EXACT_MAX_L3_ROWS` the script switches to the greedy method.

## 5. Choosing the minimum-weight basis: matroid greedy

Below, $k$ is the number of rows of the level being solved ($k=n_2$ for L2, $k=n_3$ for L3), which equals the dimension of the quotient space.

**In one sentence.** Go through the cosets from lightest to heaviest and keep only those that "cannot be produced by adding the ones already kept". This alone gives a basis with the minimum sum of weights. The reason is the following property of linear independence:

> $i$ independent vectors do not all fit in the space spanned by $i-1$ vectors (there are not enough dimensions).

Thanks to it, the typical failure of greedy methods ("taking a light one first forces a heavy one later") does not happen. The procedure, an example and the proof follow.

### 5.1 Procedure

The problem is: "each of the $2^k-1$ nonzero cosets has a weight $w(q)$; choose $k$ of them that form a basis with the minimum sum of weights."

1. Sort all cosets by increasing weight.
2. Go through them in order; keep a coset if it is linearly independent of those already kept, and discard it if it is dependent (a sum of kept ones).
3. Stop when $k$ cosets are kept.

### 5.2 Example

Take a quotient space with $k=3$ and write its 7 nonzero cosets by their coordinates (in the sense of 3.4 (c)), with the following weights.

| Coordinate | 110 | 011 | 101 | 111 | 100 | 010 | 001 |
|---|---|---|---|---|---|---|---|
| Weight | 2 | 2 | 3 | 4 | 5 | 5 | 6 |

Going from the lightest:

| Coset | Weight | Decision | Reason |
|---|---|---|---|
| 110 | 2 | keep | the first one |
| 011 | 2 | keep | a different vector from 110 |
| 101 | 3 | **discard** | $110+011=101$: a sum of kept ones |
| 111 | 4 | keep | not a sum of 110 and 011 |

The result is $\{110,\ 011,\ 111\}$ with sum of weights $2+2+4=8$.

We check directly that this is minimal. The three lightest, $110,\ 011,\ 101$, are dependent, so they cannot all be used ($2+2+3=7$ is not achievable). At most two of them can be used, and the third element has to come from those of weight 4 or more. So the sum is at least $2+2+4=8$.

**Why discarding 101 loses nothing.** Suppose some basis contains 101. Since 110, 011, 101 are dependent, that basis cannot contain both 110 and 011. Swapping 101 for the one that is missing (say 011) does not change the span, so it is still a basis, and the weight drops $3\to2$. So any basis that uses a discarded coset can be turned into one that does not, with weight no larger.

### 5.3 Why it is optimal in general (proof)

Let $g_1,g_2,\dots,g_k$ be the cosets kept by the greedy method, in the order kept (lightest first, so $w(g_1)\le\dots\le w(g_k)$). Take any other basis, sorted by increasing weight: $b_1,\dots,b_k$. We show

$$
w(g_i)\le w(b_i)\qquad(\text{for all } i) .
$$

Summing over $i$ then shows that the greedy sum of weights is no larger than that of any basis.

By contradiction, assume $w(b_i)<w(g_i)$ for some $i$.

1. $b_1,\dots,b_i$ are $i$ independent vectors, each of weight at most $w(b_i)$, so lighter than $g_i$.
2. The space spanned by $g_1,\dots,g_{i-1}$ has dimension $i-1$, so it cannot contain all of the $i$ independent vectors $b_1,\dots,b_i$. Some $b_j$ lies outside it.
3. $b_j$ is lighter than $g_i$, so the greedy method looked at $b_j$ before $g_i$. At that moment only some of $g_1,\dots,g_{i-1}$ were kept, and $b_j$ is not a sum of them. So the greedy method must have kept $b_j$.
4. Then $b_j$ is one of $g_1,\dots,g_{i-1}$, which contradicts "$b_j$ lies outside that space".

So the assumption is false and $w(g_i)\le w(b_i)$ for all $i$. The greedy solution is not only minimal in sum: sorted by weight, its first, second, … elements are each minimal compared with any basis.

On the example of 5.2: against the greedy solution $(2,2,4)$, the basis $\{110,101,111\}$ gives $(2,3,4)$ and the basis $\{100,010,001\}$ gives $(5,5,6)$; in every position the greedy solution is smaller or equal.

### 5.4 About the word "matroid"

The proof only used "$i$ independent vectors do not fit in a space of dimension $i-1$" (step 2). Restated as

> if $X$, $Y$ are independent sets with $\lvert X\rvert>\lvert Y\rvert$, some element of $X$ can be added to $Y$ keeping it independent

(the exchange axiom), together with "a subset of an independent set is independent", it defines a structure called a **matroid**. In a matroid the greedy method is known to always return a minimum-weight basis. Linear independence of vectors is the standard example (a linear matroid).

Kruskal's algorithm for the minimum spanning tree of a graph (go through the edges from lightest, keep an edge if it does not close a cycle) is another instance of the same theorem, with "does not close a cycle" in the role of "linearly independent".

### 5.5 Implementation notes

Independence is tested by GF(2) elimination on the coset index $x\in\mathbb F_2^{k}$ itself (the bits of the index are the coordinates in the quotient space). For each kept coset the lightest representative is recomputed and becomes the new row.

Doing this for L2 ($V_2/V_1$) and then L3 ($V_3/V_2$) gives, by the decomposition of section 3, the **global minimum of the total weight**. No amount of extra search finds a lighter matrix.

Cosets of equal weight, and several representatives of equal weight, are ties. The minimum total weight is unique but the matrix attaining it is not. Ties are broken with the random numbers of `SOLVE_SEED`.

## 6. Flattening: evening out the column weights at the same total weight

Stage 1 only guarantees the total weight $\sum_c w_c$; the maximum column weight $\max_c w_c$ is a by-product. So we use the freedom left by the ties.

For each row, collect **all minimum-weight representatives of its coset** ([_min_weight_reps](reduce_column_weight.py#L179)). Ties have two sources:

- a block covered exactly half ($2a_j=\lvert B_j\rvert$): the weight is the same with or without that L1 row;
- different L2 subsets $S$ that give the same minimum weight.

Replacing a row by another member of its pool does not change the coset, so the spans are preserved and so is the total weight. Only the placement of the 1s changes.

The objective is the **lexicographic minimization** of the vector of column weights sorted in decreasing order,

$$
w^{\downarrow}=(w_{(1)}\ge w_{(2)}\ge\dots\ge w_{(n)}) .
$$

It lowers the maximum first, then the number of columns that attain the maximum, and so on. A local search tries the candidates of each row in turn and accepts one when $w^{\downarrow}$ becomes strictly smaller lexicographically, repeating until nothing improves. The implementation replaces the comparison of $w^{\downarrow}$ by "comparing the histogram of weights from the top", and updates it only on the columns that changed.

## 7. Column-cap annealing

When `MAX_COLUMN_WEIGHT` $=t$ is set, the script anneals until $\max_c w_c\le t$. Here the constraint "minimum-weight coset representative" is dropped: flatness is bought by **allowing the total weight to grow a little**.

**State and neighborhood.** A state is a valid matrix $(H_2,H_3)$. One move is an elementary row operation

$$
r_i\leftarrow r_i\oplus g ,
$$

with $g$ chosen as follows.

| Level of $r_i$ | Allowed $g$ |
|---|---|
| L2 | an L1 row or another L2 row, **in the same half of the columns as $r_i$** |
| L3 | an L1 row, an L2 row, or another L3 row |

Every move is a block-lower-triangular elementary matrix, so every state visited is valid in the sense of section 2. The same-half restriction on L2 keeps the L2 block in the form $[[M_1, O],[O, M_2]]$ of our construction (each L2 row is supported on one of the two copies of the inner code); the script orders the L2 rows left half first and checks this form before saving.

**Energy.** The sum of squared excesses over the cap:

$$
E=\sum_{c=1}^{n}\max(0,\ w_c-t)^2
$$

$E=0\iff\max_c w_c\le t$. One move only changes the columns of $\mathrm{supp}(g)$, and each column weight moves by $\pm1$, so $\Delta E$ is computed from $\mathrm{supp}(g)$ alone.

**Acceptance.** Metropolis: accept if $\Delta E\le0$, otherwise with probability $e^{-\Delta E/T}$. The temperature decreases geometrically from $T=3.0$ to $0.02$.

**Choosing moves.** 70% of the moves are targeted: pick an over-cap column $c$, a row $r_i$ covering $c$, and an allowed generator $g$ that also covers $c$. The XOR then removes exactly one 1 from column $c$. The remaining 30% are uniformly random moves, for diversity.

**Multiple chains and restarts.** `ANNEAL_SEEDS` independent chains run in parallel, and the pass ends as soon as one reaches $E=0$. If none does, every chain restarts from the best state found (up to `ANNEAL_REPEATS` times).

**Polish.** After reaching $E=0$, a greedy post-pass accepts random elementary row operations only when "no column exceeds $t$ and the total weight strictly decreases", winning back part of the total weight the annealing added.

**Lower bound.** The maximum is at least the mean, so

$$
\max_c w_c\ \ge\ \Bigl\lceil \frac{W}{n}\Bigr\rceil\ \ge\ \Bigl\lceil \frac{W_{\min}}{n}\Bigr\rceil .
$$

$W_{\min}$ is the exact minimum of stage 1, so no valid matrix achieves a $t$ below the right-hand side. The script prints this bound at the start of the annealing. Whether the gap between the bound and the value actually reached can be closed is not proved by this method.

## 8. Greedy fallback (when L3 has many rows)

A local search for when the $2^{n_3}$ enumeration is infeasible. In each round:

- for each L2 row $r_i$: try $\mathrm{fold}(r_i\oplus x)$ for every subset sum $x$ of the other L2 rows and replace the row by the lightest;
- for each L3 row $r_i$: add combinations of up to `MAX_L3_COMBO` other L3 rows × every subset of the L2 rows, fold, and replace the row by the lightest.

The coefficient of the row itself always stays 1, so the transformation is invertible. The search stops when nothing improves (or after `ROUNDS` rounds); the result is a local optimum with no guarantee of global optimality. After convergence, more rounds do not change the result.

## 9. Summary

- The information-preserving transformations are "invertible recombination within a level + adding lower levels", the same as the transformations that preserve the flag $V_1\subset V_2\subset V_3$.
- Within them, minimizing the total weight decomposes into minimum-weight basis problems on the quotient spaces $V_2/V_1$ and $V_3/V_2$.
- Coset weights are computed exactly by the closed-form fold "add an L1 row when more than half of its block is covered" (the L1 supports are disjoint) and a full enumeration of the L2 subsets.
- The minimum-weight basis is found exactly by the matroid greedy method. Up to here the result is a proved optimum.
- The maximum column weight is a different objective, lowered further by flattening on ties (total weight unchanged) and by annealing (a little total weight is sacrificed). These are heuristics, and a gap to the lower bound $\lceil W_{\min}/n\rceil$ can remain.
- The objective is only the total number of 1s and the column weights; the row weight of individual checks is not constrained.
