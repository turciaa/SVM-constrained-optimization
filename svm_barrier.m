function [alpha, history] = svm_barrier(Q, y, C, tol, max_iter)
% SVM_BARRIER  Algorithm 4: interior point (logarithmic barrier).
%
% Barrier problem (for a fixed mu > 0):
%   min  phi_mu(a) = 0.5 a'Q a - 1'a - mu * sum( log(a_i) + log(C - a_i) )
%   s.t. y'a = 0
%
% Gradient:
%   grad phi_mu = Q a - 1 - mu./a + mu./(C - a)
% Hessian:
%   H = Q + diag( mu./a.^2 + mu./(C - a).^2 )    <- D > 0 so H is STRICTLY PD
%
% KKT system (equality y'a = 0):
%   [ H   y ] [ da   ]   [ -grad ]
%   [ y'  0 ] [ dnu  ] = [ -y'a  ]
%
% Key difference vs. projected Newton:
%   - Here the Hessian H = Q + diag(...) DEPENDS ON a => it must be
%     REFACTORIZED at every Newton iteration (O(n^3) per step).
%   - The barrier moves the inequalities into the objective (no projection needed).
%   - Outer scheme: decrease mu -> the central path converges to the exact solution.
%
% Stopping: on the central path the duality gap is m*mu with m = 2n
% inequalities, so we stop once 2n*mu < tol. max_iter caps the TOTAL number
% of inner Newton iterations.
%
% Numerical care:
%   - alpha_i must stay STRICTLY inside (0, C). We use "step to boundary"
%     with factor 0.99 so we never hit the bounds (log diverges, D blows up).

    n = size(Q, 1);

    % --- Strictly interior initialization with y'alpha = 0 ---
    % Use c+ for class +1 and c- for class -1 such that n+*c+ = n-*c- and 0<c<C.
    n_pos = sum(y ==  1);
    n_neg = sum(y == -1);
    alpha = zeros(n, 1);
    if n_pos >= n_neg
        alpha(y ==  1) = (n_neg / n_pos) * (C/2);
        alpha(y == -1) = C/2;
    else
        alpha(y ==  1) = C/2;
        alpha(y == -1) = (n_pos / n_neg) * (C/2);
    end

    mu          = 1.0;          % initial barrier parameter
    mu_reduce   = 0.1;          % mu <- mu/10
    % Number of reductions needed to reach a duality gap 2n*mu < tol:
    max_outer   = ceil(log(2*n*mu/tol) / log(1/mu_reduce)) + 1;
    max_inner   = 25;
    tol_inner   = 1e-6;

    history.obj       = zeros(1, max_iter);
    history.grad_norm = zeros(1, max_iter);
    history.time      = zeros(1, max_iter);
    history.iter      = 0;
    t0 = tic;
    total = 0;

    for outer = 1:max_outer
        for inner = 1:max_inner
            % Gradient and Hessian
            inv_a  = 1 ./ alpha;
            inv_ca = 1 ./ (C - alpha);
            g      = Q*alpha - 1 - mu*inv_a + mu*inv_ca;
            D      = mu*inv_a.^2 + mu*inv_ca.^2;
            H      = Q + diag(D);            % REFACTORIZED at every step!

            % KKT system
            KKT = [H, y; y', 0];
            rhs = [-g; -(y'*alpha)];
            sol = KKT \ rhs;                 % O(n^3) per iteration
            d   = sol(1:n);

            % --- Step to boundary with factor 0.99 ---
            t_max = 1.0;
            neg = d < 0;
            pos = d > 0;
            if any(neg)
                t_max = min(t_max, 0.99 * min(-alpha(neg) ./ d(neg)));
            end
            if any(pos)
                t_max = min(t_max, 0.99 * min((C - alpha(pos)) ./ d(pos)));
            end

            % --- Armijo backtracking on phi_mu ---
            phi_cur = barrier_obj(alpha, Q, mu, C);
            t       = t_max; beta = 0.5; c_arm = 1e-4;
            gtd     = g'*d;
            for ls = 1:30
                a_try = alpha + t*d;
                if all(a_try > 0) && all(a_try < C)
                    phi_try = barrier_obj(a_try, Q, mu, C);
                    if phi_try <= phi_cur + c_arm * t * gtd
                        break;
                    end
                end
                t = t * beta;
            end
            alpha = alpha + t*d;

            total = total + 1;
            history.obj(total)       = sum(alpha) - 0.5*alpha'*Q*alpha;
            history.grad_norm(total) = t * norm(d);   % actual step taken
            history.time(total)      = toc(t0);
            history.iter             = total;

            if norm(d) < tol_inner || total >= max_iter, break; end
        end

        % Outer criterion: duality gap 2n*mu (n inequalities on each side)
        if 2*n*mu < tol || total >= max_iter
            break;
        end
        mu = mu * mu_reduce;
    end

    history.obj       = history.obj(1:total);
    history.grad_norm = history.grad_norm(1:total);
    history.time      = history.time(1:total);
end

function val = barrier_obj(a, Q, mu, C)
    val = 0.5*a'*Q*a - sum(a) - mu*sum(log(a)) - mu*sum(log(C - a));
end
