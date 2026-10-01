function [acc, f1, auc] = evaluate_metrics(y_true, y_pred, scores)
% EVALUATE_METRICS  Accuracy, F1-score and ROC AUC for binary +-1 classification.
%   y_true, y_pred in {-1, +1}
%   scores : real-valued decision function values -- used for the AUC

    y_true = y_true(:);
    y_pred = y_pred(:);
    scores = scores(:);

    % Accuracy
    acc = mean(y_pred == y_true);

    % Precision / Recall / F1 (positive class = +1)
    tp = sum((y_pred ==  1) & (y_true ==  1));
    fp = sum((y_pred ==  1) & (y_true == -1));
    fn = sum((y_pred == -1) & (y_true ==  1));
    if tp + fp == 0, prec = 0; else, prec = tp/(tp+fp); end
    if tp + fn == 0, rec  = 0; else, rec  = tp/(tp+fn); end
    if (prec + rec) > 0
        f1 = 2*prec*rec/(prec + rec);
    else
        f1 = 0;
    end

    % ROC AUC via the Mann-Whitney U statistic with average ranks:
    %   AUC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    % Average ranks make the result independent of the order of tied scores.
    n_pos = sum(y_true ==  1);
    n_neg = sum(y_true == -1);
    if n_pos == 0 || n_neg == 0
        auc = 0.5;
        return;
    end
    [s, idx] = sort(scores);
    r = zeros(size(s));
    m = numel(s);
    i = 1;
    while i <= m
        j = i;
        while j < m && s(j+1) == s(i), j = j + 1; end
        r(idx(i:j)) = (i + j) / 2;              % average rank of the tie group
        i = j + 1;
    end
    auc = (sum(r(y_true == 1)) - n_pos*(n_pos+1)/2) / (n_pos*n_neg);
end
