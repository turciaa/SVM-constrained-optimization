function fig = plot_convergence(histories, names, ref_obj)
% PLOT_CONVERGENCE  Convergence comparison of the 4 algorithms.
%   - subplot(1): dual objective vs. iterations (log X axis so Newton is visible)
%   - subplot(2): step norm vs. iterations (log Y)
%   - subplot(3): error vs. CPU time (log X, log Y)
%   - subplot(4): iterations after which the error stays below given thresholds
%
% Newton (active set) needs tens of iterations, projected gradient thousands.
% To keep Newton visible on the same plot as the others we use a logarithmic
% X axis in subplots 1 and 3, plus distinct markers.

    if nargin < 3, ref_obj = []; end
    K       = numel(histories);
    colors  = lines(max(K, 4));
    markers = {'o','s','^','d','v','>'};

    fig = figure('Name','Convergence', 'Position', [60 60 1300 800]);

    %% (1) Dual objective vs. iterations (log X -- Newton and GradProj on one plot)
    subplot(2,2,1); hold on; grid on; box on;
    for i = 1:K
        iters = 1:histories{i}.iter;
        plot(iters, histories{i}.obj, '-', ...
             'LineWidth', 1.8, 'Color', colors(i,:), ...
             'Marker', markers{mod(i-1,numel(markers))+1}, ...
             'MarkerIndices', unique(round(logspace(0, log10(max(1,histories{i}.iter)), 8))), ...
             'MarkerSize', 6, 'MarkerFaceColor', colors(i,:));
    end
    if ~isempty(ref_obj)
        yline(ref_obj, '--k', 'reference', 'LabelHorizontalAlignment','left');
    end
    set(gca,'XScale','log');
    xlabel('Iteration  (log scale)'); ylabel('g(\alpha)');
    title('Dual objective vs. iterations');
    legend(names, 'Location', 'southeast');
    xlim([1 inf]);

    %% (2) Step norm vs. iterations (log Y)
    subplot(2,2,2); hold on; grid on; box on;
    for i = 1:K
        iters = 1:histories{i}.iter;
        gn    = max(histories{i}.grad_norm, 1e-16);
        semilogy(iters, gn, '-', ...
                 'LineWidth', 1.8, 'Color', colors(i,:), ...
                 'Marker', markers{mod(i-1,numel(markers))+1}, ...
                 'MarkerIndices', unique(round(logspace(0, log10(max(1,histories{i}.iter)), 8))), ...
                 'MarkerSize', 6, 'MarkerFaceColor', colors(i,:));
    end
    set(gca,'XScale','log','YScale','log');
    xlabel('Iteration  (log scale)'); ylabel('||\Delta\alpha||  (log scale)');
    title('Step norm (Newton: finite termination, Grad/ADMM: linear)');
    % ADMM records its primal + dual residual instead of a step norm
    leg2 = names;
    leg2(strcmp(leg2, 'ADMM')) = {'ADMM (r_{prim} + r_{dual})'};
    legend(leg2, 'Location', 'northeast');
    xlim([1 inf]);

    %% (3) CPU time vs. accuracy (objective error w.r.t. the reference)
    subplot(2,2,3); hold on; grid on; box on;
    if ~isempty(ref_obj)
        for i = 1:K
            err = abs(histories{i}.obj - ref_obj);
            err = max(err, 1e-16);
            tm  = max(histories{i}.time, 1e-6);
            semilogy(tm, err, '-', ...
                     'LineWidth', 1.8, 'Color', colors(i,:), ...
                     'Marker', markers{mod(i-1,numel(markers))+1}, ...
                     'MarkerIndices', unique(round(logspace(0, log10(max(1,length(err))), 8))), ...
                     'MarkerSize', 6, 'MarkerFaceColor', colors(i,:));
        end
        set(gca,'XScale','log','YScale','log');
        xlabel('CPU time (s)  [log scale]'); ylabel('|g(\alpha) - g_{ref}|  (log scale)');
        title('Time / accuracy trade-off');
        legend(names, 'Location', 'northeast');
    else
        text(0.5, 0.5, 'reference objective not available','HorizontalAlignment','center');
        axis off;
    end

    %% (4) Iterations needed for accuracy thresholds
    subplot(2,2,4); hold on; grid on; box on;
    thresh = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6];
    if ~isempty(ref_obj)
        for i = 1:K
            err = abs(histories{i}.obj - ref_obj);
            it_needed = nan(size(thresh));
            % First iteration after which the error STAYS below the threshold
            % (Barrier and ADMM are not monotone in the dual objective, so the
            % first crossing can overstate the accuracy they end up with).
            for t = 1:numel(thresh)
                last_above = find(err > thresh(t), 1, 'last');
                if isempty(last_above)
                    it_needed(t) = 1;
                elseif last_above < numel(err)
                    it_needed(t) = last_above + 1;
                end
            end
            plot(-log10(thresh), it_needed, '-', ...
                 'LineWidth', 1.8, 'Color', colors(i,:), ...
                 'Marker', markers{mod(i-1,numel(markers))+1}, ...
                 'MarkerSize', 8, 'MarkerFaceColor', colors(i,:));
        end
        set(gca,'YScale','log');
        xlabel('-log_{10}(accuracy)'); ylabel('Iterations needed  (log scale)');
        title('How fast each method reaches a given accuracy');
        legend(names, 'Location', 'northwest');
    end
end
