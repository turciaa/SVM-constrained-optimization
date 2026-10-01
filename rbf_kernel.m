function [Q, K] = rbf_kernel(X, y, gamma)
% RBF_KERNEL  Compute K(i,j) = exp(-gamma ||x_i - x_j||^2) and Q = (y y') .* K.
%   X     : n x p
%   y     : n x 1 with values +-1
%   gamma : scalar > 0
%   Outputs: Q, K (both n x n, symmetric, PSD)

    sq = sum(X.^2, 2);                % n x 1
    D2 = sq + sq' - 2*(X*X');         % squared distances (may be slightly negative numerically)
    D2 = max(D2, 0);
    K  = exp(-gamma * D2);
    K  = (K + K')/2;                  % numerical symmetrization
    Q  = (y * y') .* K;
    Q  = (Q + Q')/2;
end
