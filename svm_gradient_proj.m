function [alpha, history] = svm_gradient_proj(Q, y, C, tol, max_iter)
% SVM_GRADIENT_PROJ  Algorithm 2: projected gradient (gradient ascent on the dual).
%
% Dual:  g(a) = 1'a - 0.5 a'Q a
%   grad g = 1 - Q a
%
% Iteration:
%   a_{k+1} = Proj_F( a_k + t * (1 - Q a_k) )
%
% Fixed step: t = 1 / lambda_max(Q) -> guaranteed (sub)linear convergence for
% L-smooth concave functions; here L = lambda_max(Q).
%
% Difference vs. Newton:
%   - Does NOT use the Hessian - only gradient + projection.
%   - LINEAR convergence at best (rate 1 - 1/kappa(Q)), not superlinear.
%   - Needs many more iterations (thousands) but each one is very cheap.

    n = size(Q, 1);
    alpha = zeros(n, 1);

    % Fixed step from the Lipschitz constant of the gradient (lambda_max(Q)).
    % eigs is fast for the largest eigenvalue; alternative: norm(Q) (more expensive).
    try
        L = eigs(Q, 1, 'largestabs', 'Tolerance', 1e-4);
    catch
        L = norm(Q);                              % fallback
    end
    t_step = 1 / max(L, 1e-12);

    history.obj       = zeros(1, max_iter);
    history.grad_norm = zeros(1, max_iter);
    history.time      = zeros(1, max_iter);
    history.iter      = 0;
    t0 = tic;

    for k = 1:max_iter
        grad      = ones(n,1) - Q*alpha;          % ascent direction
        alpha_new = project_feasible(alpha + t_step * grad, y, C);

        step  = norm(alpha_new - alpha);
        alpha = alpha_new;

        history.obj(k)       = sum(alpha) - 0.5*alpha'*Q*alpha;
        history.grad_norm(k) = step;
        history.time(k)      = toc(t0);
        history.iter         = k;

        % Stop on the projected gradient norm ||a_new - a|| / t (invariant to
        % the scale of Q). Testing the step itself (step < tol) would really
        % mean ||projected gradient|| < tol * L, i.e. much looser when L >> 1.
        if step / t_step < tol, break; end
    end

    history.obj       = history.obj(1:history.iter);
    history.grad_norm = history.grad_norm(1:history.iter);
    history.time      = history.time(1:history.iter);
end
