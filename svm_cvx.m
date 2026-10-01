function [alpha, history] = svm_cvx(Q, y, C)
% SVM_CVX  Solve the SVM dual with CVX (declarative modeling).
%
% Problem:  min  0.5 alpha' Q alpha - 1'alpha
%           s.t. y'alpha = 0,  0 <= alpha_i <= C
%
% CVX is a symbolic parser for convex optimization that:
%   - recognizes the mathematical form (QP) and converts it to a standard form
%   - calls an underlying solver (SDPT3, SeDuMi, MOSEK)
%   - keeps the code very close to the math, but is slower than dedicated solvers
%
% Difference vs. quadprog/fmincon and vs. the 4 custom algorithms:
%   - quadprog: native MATLAB QP solver, very fast for QPs
%   - fmincon : general NLP solver (interior-point/SQP)
%   - CVX     : declarative parser + black-box solver (SDPT3/SeDuMi/MOSEK)
%   - custom algorithms (Newton/GradProj/ADMM/Barrier): exploit the specific
%     SVM structure (constant Hessian, separable box constraints, etc.)
%
% NOTE: requires CVX to be installed (http://cvxr.com).

    n = size(Q, 1);

    % Numerical symmetrization (CVX is strict about PSD)
    Qs = (Q + Q') / 2;

    t0 = tic;
    cvx_begin quiet
        variable a(n)
        minimize( 0.5 * quad_form(a, Qs) - sum(a) )
        subject to
            y' * a == 0;        %#ok<EQEFF>
            a >= 0;              %#ok<VUNUS>
            a <= C;              %#ok<VUNUS>
    cvx_end
    t_total = toc(t0);

    alpha = a;
    history.time      = t_total;
    history.iter      = 1;        % CVX does not expose its internal iteration count
    history.obj       = sum(alpha) - 0.5*alpha'*Qs*alpha;
    history.grad_norm = 0;
    history.status    = cvx_status;
end
