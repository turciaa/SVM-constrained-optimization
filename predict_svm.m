function [y_pred, scores, b] = predict_svm(alpha, y_train, X_train, X_test, gamma, C)
% PREDICT_SVM  Prediction with an RBF-kernel SVM from the dual solution alpha.
%
%   f(x) = sum_i alpha_i y_i K(x_i, x) + b
%
%   b is computed from the free support vectors (0 < alpha_i < C):
%     for such an i:  y_i (sum_j alpha_j y_j K(x_j, x_i) + b) = 1
%     =>  b = y_i - sum_j alpha_j y_j K(x_j, x_i)
%   and averaged over all free SVs for numerical stability.

    tol_sv = 1e-5;

    % Kernel between X_train and X_test
    sq1 = sum(X_train.^2, 2);
    sq2 = sum(X_test.^2,  2);
    D2  = sq1 + sq2' - 2*(X_train * X_test');
    D2  = max(D2, 0);
    K_te = exp(-gamma * D2);                       % n_train x n_test

    % Bias from the free SVs
    free_sv = (alpha > tol_sv) & (alpha < C - tol_sv);
    if any(free_sv)
        % K_train(:, free_sv)
        sq_fs = sum(X_train(free_sv,:).^2, 2);
        D2f   = sq1 + sq_fs' - 2*(X_train * X_train(free_sv,:)');
        D2f   = max(D2f, 0);
        K_fs  = exp(-gamma * D2f);
        b_vals = y_train(free_sv) - ((alpha .* y_train)' * K_fs)';
        b = mean(b_vals);
    else
        % Fallback: average the KKT estimate over all SVs
        sv = alpha > tol_sv;
        if any(sv)
            sq_s = sum(X_train(sv,:).^2, 2);
            D2s  = sq1 + sq_s' - 2*(X_train * X_train(sv,:)');
            D2s  = max(D2s, 0);
            K_s  = exp(-gamma * D2s);
            b_vals = y_train(sv) - ((alpha .* y_train)' * K_s)';
            b = mean(b_vals);
        else
            b = 0;
        end
    end

    scores = ((alpha .* y_train)' * K_te)' + b;
    scores = scores(:);
    y_pred = sign(scores);
    y_pred(y_pred == 0) = 1;
end
