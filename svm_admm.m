function [alpha, history] = svm_admm(Q, y, C, tol, max_iter)
% SVM_ADMM  Algorithm 3: ADMM (Alternating Direction Method of Multipliers).
%
% Reformulation:  min  f(a) + g(z)   s.t.  a = z
%   f(a) = 0.5 a'Q a - 1'a
%   g(z) = I_F(z)             (indicator function of the feasible set)
%
% Augmented Lagrangian in scaled form (u = lambda/rho):
%   L_rho(a,z,u) = f(a) + g(z) + (rho/2)||a - z + u||^2
%
% Updates:
%   a-step : (Q + rho I) a = 1 + rho (z - u)     <- factorized only once!
%   z-step : z = Proj_F( a + u )                 <- decoupled from the a-step
%   u-step : u = u + a - z                       <- multiplier update (free)
%
% Difference vs. Newton/Barrier:
%   - The objective (a) is decoupled from the constraints (z) through the
%     auxiliary variable z.
%   - The a-step is a strictly PD linear system (Q + rho I > 0) - Cholesky
%     always succeeds.
%   - The z-step only deals with the constraints (projection).
%   - Linear convergence, robust to poor conditioning of Q.

    n = size(Q, 1);
    rho = 1.0;                                    % ADMM penalty parameter

    alpha = zeros(n,1);
    z     = zeros(n,1);
    u     = zeros(n,1);

    % Cholesky of Q + rho I (strictly PD) - ONLY ONCE. O(n^3) up front, then
    % every a-step is O(n^2). This is the main advantage of ADMM here.
    R = chol(Q + rho*eye(n));                     % R'R = Q + rho I

    history.obj       = zeros(1, max_iter);
    history.grad_norm = zeros(1, max_iter);
    history.time      = zeros(1, max_iter);
    history.iter      = 0;
    t0 = tic;

    for k = 1:max_iter
        % a-step
        rhs   = ones(n,1) + rho*(z - u);
        alpha = R \ (R' \ rhs);

        % z-step (projection onto F)
        z_old = z;
        z     = project_feasible(alpha + u, y, C);

        % u-step
        u = u + alpha - z;

        % Residuals (Boyd et al. stopping criterion)
        r_prim = norm(alpha - z);                 % primal residual
        r_dual = rho * norm(z - z_old);           % dual residual

        history.obj(k)       = sum(z) - 0.5*z'*Q*z;   % z is in F => feasible objective
        history.grad_norm(k) = r_prim + r_dual;
        history.time(k)      = toc(t0);
        history.iter         = k;

        if r_prim < tol && r_dual < tol
            break;
        end
    end

    alpha = z;                                    % return the feasible point

    history.obj       = history.obj(1:history.iter);
    history.grad_norm = history.grad_norm(1:history.iter);
    history.time      = history.time(1:history.iter);
end
