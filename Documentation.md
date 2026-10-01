# Documentation

Technical documentation for the dual SVM solvers in this repository. It covers the problem formulation, each algorithm and its derivation, the shared components, every function's interface, how to read the output and the figures, and the known limitations.

For a quick overview and how to run the project, see [README.md](README.md).

---

## Contents

1. [Problem formulation](#1-problem-formulation)
2. [Pipeline overview](#2-pipeline-overview)
3. [Algorithms](#3-algorithms)
   - [3.1 Projected Newton](#31-projected-newton-svm_newtonm)
   - [3.2 Projected gradient](#32-projected-gradient-svm_gradient_projm)
   - [3.3 ADMM](#33-admm-svm_admmm)
   - [3.4 Log-barrier interior point](#34-log-barrier-interior-point-svm_barrierm)
   - [3.5 Comparison](#35-comparison)
4. [Shared components](#4-shared-components)
5. [Function reference](#5-function-reference)
6. [Output and figures](#6-output-and-figures)
7. [Verified results](#7-verified-results)
8. [Limitations and notes](#8-limitations-and-notes)

---

## 1. Problem formulation

### Primal

Given training data $(x_i, y_i)$, $i = 1..n$, with $y_i \in \{-1, +1\}$, the soft-margin kernel SVM solves

$$
\min_{w, b, \xi}\ \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i
\quad\text{s.t.}\quad y_i(w^\top\phi(x_i) + b) \ge 1 - \xi_i,\quad \xi_i \ge 0 .
$$

### Dual

All algorithms in this repository solve the dual problem, written as a minimization:

$$
\min_{\alpha \in \mathbb R^n}\ f(\alpha) = \tfrac12\,\alpha^\top Q\,\alpha - \mathbf 1^\top\alpha
\quad\text{s.t.}\quad y^\top\alpha = 0,\qquad 0 \le \alpha_i \le C .
$$

Here:

- $Q_{ij} = y_i y_j K(x_i, x_j)$, where the RBF kernel is $K(x, x') = \exp(-\gamma\lVert x - x'\rVert^2)$.
- $Q$ is symmetric positive semidefinite, so the problem is a **convex QP**. Its Hessian is the constant $Q$, and its gradient is $\nabla f(\alpha) = Q\alpha - \mathbf 1$.
- The feasible set is $F = \{\alpha : y^\top\alpha = 0,\ 0 \le \alpha \le C\}$, the intersection of a box and a hyperplane.

The printed tables and plots report the **dual objective** $g(\alpha) = -f(\alpha) = \mathbf 1^\top\alpha - \tfrac12\alpha^\top Q\alpha$, which is maximized.

### Optimality (KKT) conditions

$\alpha$ is optimal if and only if there is a scalar $\nu$ (the multiplier of $y^\top\alpha = 0$) such that the reduced gradient $r = Q\alpha - \mathbf 1 + \nu y$ satisfies:

| Component | Condition | Meaning |
|---|---|---|
| $\alpha_i = 0$ | $r_i \ge 0$ | non-support vector |
| $0 < \alpha_i < C$ | $r_i = 0$ | free support vector (on the margin) |
| $\alpha_i = C$ | $r_i \le 0$ | bounded support vector (inside the margin or misclassified) |

An equivalent, multiplier-free optimality measure is the **projected gradient**:

$$
\lVert \alpha - \Pi_F(\alpha - \nabla f(\alpha)) \rVert = 0 .
$$

`svm_newton` uses it as a stopping criterion, and `svm_gradient_proj` uses the same quantity scaled by $1/t$.

---

## 2. Pipeline overview

`main.m` runs the steps below in order.

1. **Data** (`load_data`): read `wdbc.data` (569 × 30) and map the labels to M → +1, B → −1. The data is shuffled with seed 42, split 80/20 at random (455 train, 114 test, not stratified), and z-scored with the training mean and standard deviation.
2. **Parameters:** $C = 1$, $\gamma = 1/p = 1/30$, `tol` $= 10^{-4}$.
3. **Kernel** (`rbf_kernel`): build $Q$ (455 × 455).
4. **References:**
   - `quadprog` with optimality tolerance $10^{-10}$. This solution, `alpha_qp`, is the ground truth.
   - `fmincon` (interior point) with the analytic gradient and Hessian.
   - CVX, if it is installed.
5. **Custom solvers:** Newton (`max_iter` 300), projected gradient (5000), ADMM (1000) and Barrier (200 total inner iterations).
6. **Comparison table:** objective, distance to `alpha_qp`, iterations and time.
7. **Test metrics** (`predict_svm` + `evaluate_metrics`): accuracy, F1 and AUC for every solution.
8. **Figures:** convergence (4 panels), ROC curve and the distribution of $\alpha^*$. They are saved to `results/`.

---

## 3. Algorithms

All four solvers share the same signature and return a `history` struct (see [§5](#5-function-reference)).

### 3.1 Projected Newton (`svm_newton.m`)

**Idea.** On a fixed "face" of the box, where it is known which $\alpha_i$ sit at 0 or $C$, the problem is an equality-constrained quadratic. One Newton step solves it exactly. The difficulty is identifying the right face. This implementation is a projected Newton method with an ε-active set, in the style of Bertsekas.

**Iteration $k$:**

1. **ε-active set.** With $\varepsilon_k = \min(0.01\,C,\ \lVert\alpha - \Pi_F(\alpha - \nabla f)\rVert)$:
   - $A_{lo} = \{i : \alpha_i \le \varepsilon_k,\ r_i > 0\}$
   - $A_{up} = \{i : \alpha_i \ge C - \varepsilon_k,\ r_i < 0\}$
   - Free set: $W = \overline{A_{lo} \cup A_{up}}$

   The threshold $\varepsilon_k > 0$ catches variables that are *about to* hit a bound. With $\varepsilon = 0$, those variables stay free, get clipped by the projection, and the face changes only slightly at each iteration. An earlier version of this code took 287 iterations because of this.
2. **Reduced Newton step.** Active variables are sent to their bound ($d_A = \text{bound} - \alpha_A$). Then the KKT system is solved on $W$:
   $$
   \begin{bmatrix} Q_{WW} + \delta I & y_W \\ y_W^\top & 0 \end{bmatrix}
   \begin{bmatrix} d_W \\ \nu \end{bmatrix} =
   \begin{bmatrix} -\nabla f_W - Q_{WA}d_A \\ -y^\top\alpha - y_A^\top d_A \end{bmatrix},
   \qquad \delta = 10^{-10}.
   $$
   The LU factorization is cached and recomputed only when $W$ changes. The returned $\nu$ updates the multiplier estimate.
3. **Projected-arc Armijo search.** Try $\alpha(t) = \Pi_F(\alpha + t\,d)$ for $t = 1, \tfrac12, \dots$. Accept when $\nabla f^\top(\alpha(t) - \alpha) < 0$ and $f(\alpha(t)) \le f(\alpha) + c\,\nabla f^\top(\alpha(t) - \alpha)$, with $c = 10^{-4}$. Along the arc, many bounds can become active or inactive at once.
4. **Fallback.** If the Newton arc gives no decrease within 30 halvings, take an Armijo projected-gradient step instead. This keeps the method globally convergent.
5. **Stop** when $\lVert\alpha - \Pi_F(\alpha - \nabla f)\rVert <$ `tol`.

**Behaviour.** Progress is fast but uneven while the active set is being found. Once it is correct, the next step lands *exactly* on the optimum (finite termination). On WDBC this takes 65 iterations and matches `quadprog` to $10^{-8}$.

**Cost.** $O(|W|^3)$ per refactorization and $O(n^2)$ per iteration otherwise, plus up to about 30 projections per line search.

### 3.2 Projected gradient (`svm_gradient_proj.m`)

**Iteration:**

$$
\alpha_{k+1} = \Pi_F\big(\alpha_k + t(\mathbf 1 - Q\alpha_k)\big), \qquad t = 1/L,\quad L = \lambda_{\max}(Q).
$$

$L$ is computed once with `eigs`, falling back to `norm(Q)`.

**Stopping rule:** $\lVert\alpha_{k+1} - \alpha_k\rVert / t <$ `tol`. This is the norm of the *gradient mapping*, which does not depend on the scale of $Q$. A plain test on $\lVert\Delta\alpha\rVert$ would be $L$ times looser ($L \approx 166$ here).

**Behaviour.** Converges linearly at best, at a rate of about $1 - 1/\kappa(Q)$. With $\kappa(Q) \approx 3\cdot10^5$ it does not reach `tol` within the 5000-iteration cap. Its final objective gap is $2.5\cdot10^{-4}$. Each iteration is cheap: one $O(n^2)$ product and one projection.

### 3.3 ADMM (`svm_admm.m`)

**Splitting.** $\min f(\alpha) + I_F(z)$ subject to $\alpha = z$. With the scaled multiplier $u$ and penalty $\rho = 1$:

| Step | Update | Cost |
|---|---|---|
| α-step | $(Q + \rho I)\,\alpha = \mathbf 1 + \rho(z - u)$ | $O(n^2)$, using a Cholesky factor computed **once** |
| z-step | $z = \Pi_F(\alpha + u)$ | one projection |
| u-step | $u \leftarrow u + \alpha - z$ | $O(n)$ |

**Stopping rule** (Boyd et al.): primal residual $\lVert\alpha - z\rVert <$ `tol` **and** dual residual $\rho\lVert z - z_{old}\rVert <$ `tol`. The function returns $z$, which is always feasible. The objective is recorded at $z$.

**Behaviour.** Converges linearly and is not monotone in the objective. On this problem it is the fastest method in wall-clock time (270 iterations, about 0.08 s).

### 3.4 Log-barrier interior point (`svm_barrier.m`)

**Barrier subproblem** for fixed $\mu > 0$:

$$
\min\ \phi_\mu(\alpha) = f(\alpha) - \mu\sum_i\big(\log\alpha_i + \log(C - \alpha_i)\big)
\quad\text{s.t.}\quad y^\top\alpha = 0 .
$$

Its Hessian is $H = Q + \mathrm{diag}\big(\mu/\alpha_i^2 + \mu/(C - \alpha_i)^2\big)$, which is strictly positive definite.

**Inner loop** (Newton on $\phi_\mu$, at most 25 iterations per $\mu$):

- Solve $\begin{bmatrix} H & y \\ y^\top & 0\end{bmatrix}\begin{bmatrix} d \\ \nu\end{bmatrix} = \begin{bmatrix}-\nabla\phi_\mu \\ -y^\top\alpha\end{bmatrix}$. $H$ changes every step, so this is $O(n^3)$ per iteration.
- Step to the boundary with factor 0.99, then Armijo backtracking on $\phi_\mu$.
- Stop the inner loop when $\lVert d\rVert < 10^{-6}$.

**Outer loop.** $\mu \leftarrow \mu/10$, starting from $\mu = 1$. On the central path the duality gap is $m\mu$ with $m = 2n$ inequalities, so the method stops when $2n\mu <$ `tol`. The number of outer steps is computed from this. `max_iter` caps the **total** number of inner iterations.

**Start point.** $\alpha$ is strictly interior with $y^\top\alpha = 0$: each class gets a constant value, scaled so that $n_+ c_+ = n_- c_-$.

**Behaviour.** 97 iterations and a final gap of $4\cdot10^{-5}$, consistent with the bound $2n\mu \approx 9\cdot10^{-5}$. The *dual objective* is not monotone. After each reduction of $\mu$, the first long step can come very close to the optimum (error $7\cdot10^{-6}$ at iteration 88). Re-centering then moves back toward the central path, which is the expected behaviour.

### 3.5 Comparison

| | Newton | Projected gradient | ADMM | Barrier |
|---|---|---|---|---|
| Uses the Hessian | yes (reduced) | no | yes (via $Q + \rho I$) | yes (changes every step) |
| Factorizations | when $W$ changes | none | 1 | every iteration |
| Handles the box by | active set + projection | projection | projection (z-step) | barrier |
| Convergence | finite once the face is found | linear | linear | superlinear inner, geometric in $\mu$ |
| Monotone dual objective | yes | yes | no | no |
| WDBC iterations / time | 65 / 0.14 s | 5000 (cap) / 0.57 s | 270 / 0.08 s | 97 / 0.36 s |

---

## 4. Shared components

### Projection onto $F$ (`project_feasible.m`)

Projecting a point $v$ means solving $\min \tfrac12\lVert a - v\rVert^2$ over $F$. Its KKT conditions give

$$
a(\lambda) = \mathrm{clip}(v - \lambda y,\ 0,\ C),
$$

where $\lambda$ is the multiplier of $y^\top a = 0$. The function $h(\lambda) = y^\top a(\lambda)$ is continuous, piecewise linear and non-increasing. The breakpoints $v_i/y_i$ and $(v_i - C)/y_i$ bracket its root:

- $h(\lambda_{lo}) = C\,n_+ \ge 0$
- $h(\lambda_{hi}) = -C\,n_- \le 0$

**Algorithm:**

1. Bisect on $\lambda$ until the bracket reaches relative width $10^{-15}$ (at most 200 steps).
2. Take the free set $\{0 < a_i < C\}$ at the bisection result.
3. Solve for $\lambda$ in closed form on that free set:
   $$
   \lambda = \frac{\sum_{\text{free}} y_i v_i + C\sum_{\text{upper}} y_i}{\sum_{\text{free}} y_i^2}
   $$

After the closed-form step, $y^\top a = 0$ holds to rounding error ($10^{-15}$). In tests the result matches a `quadprog` projection to $10^{-10}$.

This replaced an earlier Dykstra alternating-projection scheme. The new version is exact, has no inner tolerance, and made Newton, projected gradient and ADMM 2–3× faster.

### Bias and prediction (`predict_svm.m`)

The decision function is

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x) + b .
$$

For every free support vector ($10^{-5} < \alpha_i < C - 10^{-5}$), the margin condition gives $b = y_i - \sum_j \alpha_j y_j K(x_j, x_i)$. The code averages this over all free SVs. If there are none, it averages over all SVs; if there are no SVs at all, $b = 0$. The prediction is $\mathrm{sign}(f(x))$, and a score of 0 maps to +1.

### Metrics (`evaluate_metrics.m`)

- **Accuracy** and **F1**: the positive class is +1 (malignant).
- **ROC AUC**: computed with the Mann–Whitney statistic and average ranks,
  $$
  \text{AUC} = \frac{\sum_{i:\,y_i = +1} \mathrm{rank}(s_i) - n_+(n_+ + 1)/2}{n_+ n_-},
  $$
  which equals $P(s_+ > s_-) + \tfrac12 P(s_+ = s_-)$. It is therefore correct when scores are tied: all-tied scores give 0.5.

---

## 5. Function reference

### Solvers

```matlab
[alpha, history] = svm_newton(Q, y, C, tol, max_iter)          % defaults: tol 1e-6, max_iter 100
[alpha, history] = svm_gradient_proj(Q, y, C, tol, max_iter)
[alpha, history] = svm_admm(Q, y, C, tol, max_iter)
[alpha, history] = svm_barrier(Q, y, C, tol, max_iter)         % max_iter = total inner iterations
[alpha, history] = svm_cvx(Q, y, C)                            % requires CVX
```

| Input | Description |
|---|---|
| `Q` | $n \times n$ matrix $(yy^\top) \circ K$ |
| `y` | $n \times 1$ labels in $\{-1, +1\}$ |
| `C` | box upper bound (> 0) |
| `tol` | stopping tolerance (see each algorithm's stopping rule) |
| `max_iter` | iteration cap |

| `history` field | Description |
|---|---|
| `iter` | number of iterations performed |
| `obj` | `1 × iter`: dual objective $g(\alpha)$ after each iteration |
| `grad_norm` | `1 × iter`: step norm $\lVert\Delta\alpha\rVert$. For ADMM it is the primal + dual residual. |
| `time` | `1 × iter`: elapsed time since the solver started (s) |
| `status` | CVX only: the solver status string |

### Utilities

```matlab
[X_train, y_train, X_test, y_test] = load_data(data_path)      % default 'wdbc.data'; synthetic fallback
[Q, K]                             = rbf_kernel(X, y, gamma)
a                                  = project_feasible(v, y, C)
[y_pred, scores, b]                = predict_svm(alpha, y_train, X_train, X_test, gamma, C)
[acc, f1, auc]                     = evaluate_metrics(y_true, y_pred, scores)
fig                                = plot_convergence(histories, names, ref_obj)
```

---

## 6. Output and figures

### Console

1. Data and $Q$ construction info.
2. Reference objectives (`quadprog`, `fmincon`, CVX).
3. **Solver table:** dual objective, $\lVert\alpha - \alpha_{qp}\rVert$, iterations and time.
4. **Test table:** accuracy, F1 and AUC per solver.

### `results/convergence.png`

| Panel | What it shows | How to read it |
|---|---|---|
| Top left: dual objective vs. iteration (log x) | $g(\alpha_k)$ for each method, with the dashed `quadprog` reference | All curves should approach the dashed line. Barrier starts far below it because it starts in the interior. ADMM may dip early because it is not monotone. |
| Top right: step norm (log-log) | $\lVert\Delta\alpha\rVert$; ADMM shows $r_{prim} + r_{dual}$ | GradProj and ADMM decrease steadily (linear rate). Barrier has a spike after every reduction of $\mu$. Newton's step is still large on its final iteration because that step lands exactly on the optimum. |
| Bottom left: error vs. CPU time (log-log) | $\lvert g(\alpha_k) - g_{ref}\rvert$ | Lower and further left is better. Newton's vertical drop is the finite termination. |
| Bottom right: iterations to accuracy | the iteration after which the error **stays** below $10^{-1}, \dots, 10^{-6}$ | A missing point means that accuracy was not held at the end. GradProj stops at $10^{-3}$ and Barrier at $10^{-4}$. |

### `results/roc_curve.png`

ROC curve of the `quadprog` solution on the test set; the title shows the AUC.

### `results/alpha_distribution.png`

Histogram of Newton's $\alpha^*$ with the counts of free SVs, bounded SVs ($\alpha = C$) and zeros.

---

## 7. Verified results

These numbers come from running `main.m` on MATLAB R2025b (Windows 11), with the setup in [§2](#2-pipeline-overview). Times depend on the machine.

| Solver | $g(\alpha)$ | $\lvert g - g_{qp}\rvert$ | $\lVert\alpha - \alpha_{qp}\rVert$ | Projected-gradient norm | Iterations | Time |
|---|---|---|---|---|---|---|
| quadprog | 54.399950 | — | — | — | — | 0.18 s |
| fmincon | 54.399944 | 6.5e-06 | 2.4e-05 | — | — | 1.3 s |
| Newton | 54.399950 | 9.6e-12 | 9.4e-09 | 2.5e-12 | 65 | 0.14 s |
| GradProj | 54.399702 | 2.5e-04 | 8.8e-02 | 6.0e-03 | 5000 (cap) | 0.57 s |
| ADMM | 54.399950 | 5.0e-07 | 7.3e-04 | 2.3e-03 | 270 | 0.08 s |
| Barrier | 54.399910 | 4.0e-05 | 1.3e-04 | 9.1e-05 | 97 | 0.36 s |

**Consistency checks (all pass):**

- Every solution is feasible: $\lvert y^\top\alpha\rvert \le 2\cdot10^{-15}$ and there are no box violations.
- No solver's objective exceeds the `quadprog` optimum by more than $10^{-11}$. A larger excess would be impossible for a feasible point and would point to a feasibility bug. Newton's $+9.6\cdot10^{-12}$ is within the reference's own precision.
- Every `history` vector has length `iter`, and `time` is non-decreasing.
- **Test set:** confusion matrix [TP 37, FN 1; FP 0, TN 76], which gives accuracy 113/114 = 0.9912, precision 1, recall 37/38 and F1 0.9867. All solvers produce the same predictions.
- The AUC from the plotted ROC curve (trapezoid rule) is 0.997922, the same as the metric. The single misclassified malignant sample scores below 6 of the 76 benign samples, so $1 - 6/(38 \cdot 76) = 0.99792$.
- Support vectors: 51 free, 57 bounded and 347 zero, which adds up to 455.

---

## 8. Limitations and notes

- **Projected gradient** doesn't converge to `tol` $= 10^{-4}$ within 5000 iterations on this problem. Accelerated variants such as Nesterov/FISTA or Barzilai–Borwein steps would fix that. Its classifier is still identical to the others.
- **Newton on ill-conditioned problems.** For very small $\gamma$ (e.g. $0.1/p$) combined with $C \ge 1$, $Q$ is nearly singular ($\kappa \approx 3\cdot10^8$). Identifying the active set then takes far longer, and the method can hit `max_iter`. Barrier stays accurate over the whole tested range $\gamma \in \{0.1, 1, 3\}/p$, $C \in \{0.1, 1, 10, 100\}$.
- **Barrier accuracy** is limited by the final $\mu$: the gap is about $2n\mu < $ `tol`. Its $\alpha$ values approach the bounds without reaching them exactly ($\min\alpha \approx 6\cdot10^{-8}$).
- **Memory:** every method forms the dense $n \times n$ matrix $Q$, and Newton and Barrier factor dense systems. This works for a few thousand samples. Larger datasets would need decomposition methods such as SMO.
- **Data split:** the split is random, not stratified. The test set has 38 malignant and 76 benign samples, compared with about 37% malignant overall.
- The `quadprog` reference is accurate to about $10^{-11}$ in objective value, so a Newton gap of about $10^{-11}$ is at the reference's own precision.
