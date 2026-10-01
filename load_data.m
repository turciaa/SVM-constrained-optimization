function [X_train, y_train, X_test, y_test] = load_data(data_path)
% LOAD_DATA  Load Breast Cancer Wisconsin (Diagnostic) from UCI, or generate
% synthetic data if the file is missing.
%   - random (non-stratified) 80/20 train/test split after a seeded shuffle
%   - z-score normalization using the training-set mean / std only
%
%   wdbc.data format (UCI):
%     col 1    : ID
%     col 2    : 'M' (malignant) / 'B' (benign)
%     col 3-32 : 30 numeric features
    if nargin < 1, data_path = 'wdbc.data'; end

    if isfile(data_path)
        fprintf('Loading WDBC from %s\n', data_path);
        T = readtable(data_path, 'FileType','text', 'ReadVariableNames', false);
        labels = T{:, 2};
        if iscell(labels) || isstring(labels)
            y = double(strcmp(labels, 'M'))*2 - 1;     % M = +1, B = -1
        else
            y = double(labels);
        end
        X = T{:, 3:end};
    else
        fprintf('wdbc.data not found -> generating synthetic data (30D, 570 samples)\n');
        rng(42);
        nc = 285; p = 30;
        mu1 = -0.6*ones(1,p);
        mu2 =  0.6*ones(1,p);
        X1  = randn(nc, p) + mu1;
        X2  = randn(nc, p) + mu2;
        X   = [X1; X2];
        y   = [-ones(nc,1); ones(nc,1)];
    end

    % Reproducible shuffle
    rng(42);
    n = size(X,1);
    perm = randperm(n);
    X = X(perm,:); y = y(perm);

    % 80/20 split
    ntr = round(0.8 * n);
    X_train = X(1:ntr,:);   y_train = y(1:ntr);
    X_test  = X(ntr+1:end,:); y_test = y(ntr+1:end);

    % Z-score using training statistics only
    mu_tr  = mean(X_train, 1);
    sig_tr = std(X_train, [], 1);
    sig_tr(sig_tr == 0) = 1;
    X_train = (X_train - mu_tr) ./ sig_tr;
    X_test  = (X_test  - mu_tr) ./ sig_tr;
end
