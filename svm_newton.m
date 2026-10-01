function [alpha, history] = svm_newton(Q, y, C, tol, max_iter)
% SVM_NEWTON  Algorithm 1: projected Newton with an epsilon-active set (Bertsekas).
%
% Minimize f(a) = 0.5 a'Q a - 1'a    s.t.  y'a = 0,  0 <= a <= C.
%   grad f = Q a - 1
%   hess f = Q             (CONSTANT)
%
% Each iteration:
%   1. The epsilon-active set A = components CLOSE to a bound (within eps_k)
%      that the reduced gradient r = g + nu*y pushes towards that bound:
%        a_i <= eps_k and r_i > 0     or     a_i >= C - eps_k and r_i < 0
%      with eps_k = min(eps0, ||projected gradient||). The rest is the free set W.
%      (The threshold eps_k > 0 anticipates the bounds: without it, variables
%      heading to 0 stay free, the Newton step clips them through the
%      projection, and the method only changes the active face a little at
%      each iteration.)
%   2. Variables in A are sent to their bound (d_A = bound - a_A), and on W
%      we take the REDUCED Newton step:
%        [Q_WW  y_W] [d_W]   [-g_W - Q_WA d_A ]
%        [y_W'   0 ] [nu ] = [-y'a - y_A' d_A ]
%   3. Armijo search along the projected arc a(t) = Proj_F(a + t d), which can
%      add / remove many active bounds in a single step.
%      If the Newton arc gives no decrease -> projected gradient step.
%
% Why the active set: a Newton step on ALL variables followed by a projection
% onto F is no longer a Newton step once some a_i sit on a bound -> the
% convergence degrades to linear. On the correct active face f is quadratic,
% so the reduced Newton step lands exactly at the minimum of that face ->
% once the active face is identified the method terminates.
%
% Key difference vs. Barrier:
%   Here the reduced KKT system is refactorized ONLY when W changes
%   (O(|W|^3), |W| = number of free variables); once the active face is
%   identified the factorization is reused. In Barrier, H = Q + diag(...)
%   changes at every step -> refactorization at every iteration.
%
% Stopping criterion: projected gradient norm ||a - Proj_F(a - grad f)|| < tol
% (exactly zero at KKT points).

    if nargin < 4, tol = 1e-6;      end
    if nargin < 5, max_iter = 100;  end

    n     = size(Q, 1);
    alpha = zeros(n, 1);                          % y'*0 = 0 => feasible
    reg   = 1e-10;                                % numerical regularization of Q_WW
    eps0  = 0.01 * C;                             % max "close to a bound" threshold
    beta  = 0.5; c_arm = 1e-4;

    g  = Q*alpha - 1;
    nu = -(y'*g) / (y'*y);                        % initial multiplier estimate
    pg = norm(alpha - project_feasible(alpha - g, y, C));
    W_prev = false(n, 1); have_fact = false;

    history.obj       = zeros(1, max_iter);
    history.grad_norm = zeros(1, max_iter);
    history.time      = zeros(1, max_iter);
    history.iter      = 0;
    t0 = tic;

    for k = 1:max_iter
        f_cur = 0.5*alpha'*Q*alpha - sum(alpha);

        % --- 1. Epsilon-active / free sets ---
        eps_k = min(eps0, pg);
        r     = g + nu*y;
        at_lo = (alpha <= eps_k)     & (r > 0);
        at_up = (alpha >= C - eps_k) & (r < 0);
        W     = ~(at_lo | at_up);
        nW    = sum(W);

        % --- 2. Reduced Newton step (refactorize only if W changed) ---
        d = zeros(n, 1);
        d(at_lo) = -alpha(at_lo);                 % sent to their bound
        d(at_up) = C - alpha(at_up);
        A = ~W;
        if nW > 0
            if ~have_fact || ~isequal(W, W_prev)
                KKT = [Q(W,W) + reg*eye(nW), y(W); y(W)', 0];
                [Lf, Uf, Pf] = lu(KKT);
                W_prev = W; have_fact = true;
            end
            rhs  = [-g(W) - Q(W,A)*d(A); -(y'*alpha) - y(A)'*d(A)];
            sol  = Uf \ (Lf \ (Pf*rhs));
            d(W) = sol(1:nW);
            nu   = sol(end);                      % multiplier at a + d
        end

        % --- 3. Armijo along the projected arc a(t) = Proj_F(a + t d) ---
        accepted = false;
        if nW > 0
            t = 1.0;
            for ls = 1:30
                a_try = project_feasible(alpha + t*d, y, C);
                dec   = g' * (a_try - alpha);
                f_try = 0.5*a_try'*Q*a_try - sum(a_try);
                if dec < 0 && f_try <= f_cur + c_arm*dec
                    accepted = true; break;
                end
                t = t * beta;
            end
        end

        % Fallback: projected gradient with Armijo (guaranteed decrease)
        if ~accepted
            s = 1.0;
            for ls = 1:40
                a_try = project_feasible(alpha - s*g, y, C);
                dec   = g' * (a_try - alpha);
                f_try = 0.5*a_try'*Q*a_try - sum(a_try);
                if f_try <= f_cur + c_arm*dec
                    break;
                end
                s = s * beta;
            end
        end

        step  = norm(a_try - alpha);
        alpha = a_try;
        g     = Q*alpha - 1;

        history.obj(k)       = sum(alpha) - 0.5*alpha'*Q*alpha;
        history.grad_norm(k) = step;
        history.time(k)      = toc(t0);
        history.iter         = k;

        % Projected gradient (stationarity measure on F)
        pg = norm(alpha - project_feasible(alpha - g, y, C));
        if pg < tol, break; end
    end

    K = history.iter;
    history.obj       = history.obj(1:K);
    history.grad_norm = history.grad_norm(1:K);
    history.time      = history.time(1:K);
end
