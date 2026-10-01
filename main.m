%% main.m  -- Dual SVM with an RBF kernel: 4 constrained optimization algorithms
% Problem:   max  g(alpha) = 1'alpha - 0.5 alpha' Q alpha
%            s.t. y'alpha = 0,   0 <= alpha_i <= C
% Equivalent: min f(alpha) = 0.5 alpha' Q alpha - 1'alpha  with the same constraints.

clc; clear; close all;
rng(42);

%% 1. Data
% Put 'wdbc.data' (UCI Breast Cancer Wisconsin) in the current folder.
% If it is missing, synthetic data is generated (two 30-D Gaussian blobs).
data_path = 'wdbc.data';
[X_train, y_train, X_test, y_test] = load_data(data_path);
[n, p] = size(X_train);
fprintf('Training: n=%d, p=%d (test: %d)\n', n, p, length(y_test));

%% 2. SVM parameters
C      = 1.0;
gamma  = 1/p;        % classic RBF heuristic
tol    = 1e-4;
maxit  = 1000;

%% 3. Kernel matrix Q = (y y') .* K
fprintf('\nBuilding the matrix Q (n x n)...\n');
tic;
[Q, K_train] = rbf_kernel(X_train, y_train, gamma);
fprintf('Q built in %.3fs (size %dx%d)\n', toc, n, n);

%% 4. Reference: quadprog (MATLAB Optimization Toolbox)
%   quadprog solves  min 0.5 x'Hx + f'x  s.t. ...  => H = Q, f = -1.
fprintf('\n--- Reference: quadprog ---\n');
opts_qp = optimoptions('quadprog','Display','off','OptimalityTolerance',1e-10);
t0 = tic;
alpha_qp = quadprog(Q, -ones(n,1), [], [], y_train', 0, ...
                    zeros(n,1), C*ones(n,1), [], opts_qp);
time_qp = toc(t0);
obj_qp = sum(alpha_qp) - 0.5*alpha_qp'*Q*alpha_qp;
fprintf('quadprog: obj* = %.6f, time = %.3fs\n', obj_qp, time_qp);

%% 5. Secondary reference: fmincon (interior-point)
fprintf('\n--- Secondary reference: fmincon (interior-point) ---\n');
opts_fm = optimoptions('fmincon','Algorithm','interior-point', ...
            'Display','off','SpecifyObjectiveGradient',true, ...
            'HessianFcn',@(a,lambda) Q, ...
            'OptimalityTolerance',1e-8,'MaxIterations',500);
t0 = tic;
alpha_fm = fmincon(@(a) neg_dual(a, Q), (C/2)*ones(n,1), [], [], y_train', 0, ...
                   zeros(n,1), C*ones(n,1), [], opts_fm);
time_fm = toc(t0);
obj_fm = sum(alpha_fm) - 0.5*alpha_fm'*Q*alpha_fm;
fprintf('fmincon : obj* = %.6f, time = %.3fs\n', obj_fm, time_fm);

%% 5b. Third reference: CVX (optional -- only if installed)
fprintf('\n--- Third reference: CVX (optional) ---\n');
cvx_ok = false;
if exist('cvx_begin','file') == 2 || exist('cvx_begin','file') == 6
    try
        [alpha_cvx, hist_cvx] = svm_cvx(Q, y_train, C);
        obj_cvx  = sum(alpha_cvx) - 0.5*alpha_cvx'*Q*alpha_cvx;
        time_cvx = hist_cvx.time;
        err_cvx  = norm(alpha_cvx - alpha_qp);
        cvx_ok   = true;
        fprintf('CVX     : obj* = %.6f, time = %.3fs, status = %s\n', ...
                obj_cvx, time_cvx, hist_cvx.status);
    catch ME
        fprintf('CVX failed: %s\n', ME.message);
    end
else
    fprintf('CVX is not installed (skipped). Get it from http://cvxr.com.\n');
end

%% 6. The 4 custom algorithms
algos = struct( ...
   'Newton',    @() svm_newton(Q, y_train, C, tol, 300), ...
   'GradProj',  @() svm_gradient_proj(Q, y_train, C, tol, 5000), ...
   'ADMM',      @() svm_admm(Q, y_train, C, tol, maxit), ...
   'Barrier',   @() svm_barrier(Q, y_train, C, tol, 200));

names = fieldnames(algos);
results = struct();

fprintf('\n--- Custom algorithms ---\n');
for i = 1:numel(names)
    nm = names{i};
    fprintf('  Running %s ...\n', nm);
    [a, hist] = algos.(nm)();
    obj   = sum(a) - 0.5*a'*Q*a;
    err   = norm(a - alpha_qp);
    results.(nm).alpha   = a;
    results.(nm).hist    = hist;
    results.(nm).obj     = obj;
    results.(nm).err     = err;
    if hist.iter > 0
        results.(nm).time = hist.time(end);
    else
        results.(nm).time = 0;                    % stopped before the first iteration
    end
    results.(nm).iter    = hist.iter;
end

%% 7. Comparison table
fprintf('\n%-12s | %-10s | %-10s | %-8s | %-8s\n', ...
        'Algorithm','obj*','||a-a_qp||','iter','time[s]');
fprintf('%s\n', repmat('-',1,60));
fprintf('%-12s | %-10.6f | %-10.2e | %-8s | %-8.3f\n', ...
        'quadprog', obj_qp, 0, 'N/A', time_qp);
fprintf('%-12s | %-10.6f | %-10.2e | %-8s | %-8.3f\n', ...
        'fmincon', obj_fm, norm(alpha_fm - alpha_qp), 'N/A', time_fm);
if cvx_ok
    fprintf('%-12s | %-10.6f | %-10.2e | %-8s | %-8.3f\n', ...
            'CVX', obj_cvx, err_cvx, 'N/A', time_cvx);
end
for i = 1:numel(names)
    nm = names{i};
    r = results.(nm);
    fprintf('%-12s | %-10.6f | %-10.2e | %-8d | %-8.3f\n', ...
            nm, r.obj, r.err, r.iter, r.time);
end

%% 8. Test-set performance
fprintf('\n%-12s | %-7s | %-7s | %-7s\n','Algorithm','Acc','F1','AUC');
fprintf('%s\n', repmat('-',1,42));
for i = 1:numel(names)
    nm = names{i};
    a  = results.(nm).alpha;
    [yp, sc] = predict_svm(a, y_train, X_train, X_test, gamma, C);
    [acc, f1, auc] = evaluate_metrics(y_test, yp, sc);
    results.(nm).acc = acc; results.(nm).f1 = f1; results.(nm).auc = auc;
    results.(nm).scores = sc;
    fprintf('%-12s | %-7.4f | %-7.4f | %-7.4f\n', nm, acc, f1, auc);
end

% quadprog too (used for the ROC plot)
[yp_qp, sc_qp] = predict_svm(alpha_qp, y_train, X_train, X_test, gamma, C);
[acc_qp, f1_qp, auc_qp] = evaluate_metrics(y_test, yp_qp, sc_qp);
fprintf('%-12s | %-7.4f | %-7.4f | %-7.4f\n','quadprog',acc_qp,f1_qp,auc_qp);

% and CVX (if it ran)
if cvx_ok
    [yp_cvx, sc_cvx] = predict_svm(alpha_cvx, y_train, X_train, X_test, gamma, C);
    [acc_cvx, f1_cvx, auc_cvx] = evaluate_metrics(y_test, yp_cvx, sc_cvx);
    fprintf('%-12s | %-7.4f | %-7.4f | %-7.4f\n','CVX',acc_cvx,f1_cvx,auc_cvx);
end

%% 9. Plots
histories = cell(1, numel(names));
for i = 1:numel(names), histories{i} = results.(names{i}).hist; end
fig_conv = plot_convergence(histories, names, obj_qp);

% ROC curve of the quadprog (reference) solution
fig_roc = figure('Name','ROC curve');
[~, idx] = sort(sc_qp, 'descend');
ys = y_test(idx);
npos = sum(y_test==1); nneg = sum(y_test==-1);
tpr = [0; cumsum(ys==1)/max(npos,1)];
fpr = [0; cumsum(ys==-1)/max(nneg,1)];
plot(fpr, tpr, 'LineWidth', 2); hold on;
plot([0 1],[0 1],'k--');
xlabel('FPR'); ylabel('TPR');
title(sprintf('ROC curve (AUC = %.4f)', auc_qp));
grid on; axis square;

% Distribution of alpha*
fig_alpha = figure('Name','Alpha distribution');
nm = 'Newton';
a  = results.(nm).alpha;
histogram(a, 40); xlabel('\alpha_i'); ylabel('count');
title(sprintf('Distribution of \\alpha^* (%s) -- free SV=%d, bounded SV=%d, zero=%d', ...
  nm, sum(a>1e-5 & a<C-1e-5), sum(a>=C-1e-5), sum(a<=1e-5)));

%% 10. Save the figures to results/
out_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~isfolder(out_dir), mkdir(out_dir); end
figs  = {fig_conv, fig_roc, fig_alpha};
files = {'convergence.png', 'roc_curve.png', 'alpha_distribution.png'};
for i = 1:numel(figs)
    % Light theme for the saved images (R2025a+ follows the desktop theme).
    if isprop(figs{i}, 'Theme'), figs{i}.Theme = 'light'; end
    ax = findall(figs{i}, 'Type', 'axes');
    for j = 1:numel(ax), ax(j).Toolbar.Visible = 'off'; end
    exportgraphics(figs{i}, fullfile(out_dir, files{i}), 'Resolution', 150);
end
fprintf('\nFigures saved to %s\n', out_dir);

%% Local functions
function [f, g] = neg_dual(a, Q)
% Negated dual objective and its gradient (fmincon minimizes).
    Qa = Q*a;
    f  = 0.5*a'*Qa - sum(a);
    g  = Qa - 1;
end
