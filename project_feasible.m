function a = project_feasible(v, y, C)
% PROJECT_FEASIBLE  Exact Euclidean projection onto
%   F = { a : y'a = 0,  0 <= a_i <= C },   y_i in {-1, +1}.
%
% The KKT conditions of  min 0.5||a - v||^2  s.t.  a in F  give
%   a(lambda) = clip(v - lambda*y, 0, C)
% for a scalar multiplier lambda of the equality constraint. The function
% h(lambda) = y'a(lambda) is continuous, piecewise linear and non-increasing,
% so its root is bracketed by the breakpoints and found by bisection. A final
% closed-form solve on the free components makes y'a = 0 hold to rounding error.
%
% (This replaces an earlier Dykstra alternating-projection scheme: the
% bisection is exact, costs O(n) per step and has no inner tolerance to tune.)

    v = v(:);  y = y(:);
    clip = @(lam) min(max(v - lam*y, 0), C);

    % Component i is strictly inside (0, C) for lambda between v_i/y_i and
    % (v_i - C)/y_i. Outside all breakpoints h(lo) = C*n_pos >= 0 and
    % h(hi) = -C*n_neg <= 0, so [lo, hi] brackets the root.
    bp = [v ./ y; (v - C) ./ y];
    lo = min(bp);
    hi = max(bp);

    for it = 1:200
        mid = 0.5*(lo + hi);
        if y' * clip(mid) > 0
            lo = mid;
        else
            hi = mid;
        end
        if hi - lo <= 1e-15 * max(1, abs(mid)), break; end
    end

    % Exact multiplier for the free set identified by the bisection:
    %   sum_free y_i (v_i - lambda y_i) + C * sum_upper y_i = 0
    lam = 0.5*(lo + hi);
    a   = clip(lam);
    fr  = (a > 0) & (a < C);
    if any(fr)
        lam = (y(fr)'*v(fr) + C*sum(y(a >= C))) / (y(fr)'*y(fr));
        a   = clip(lam);
    end
end
