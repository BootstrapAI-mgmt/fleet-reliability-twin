function out = availability_mc(U, P, usage, H, R, tat, seed)
%AVAILABILITY_MC  Fleet availability and failure-count forecast by Monte Carlo.
%
%   out = AVAILABILITY_MC(U, P, usage, H, R, tat, seed)
%
%   U : M x 5  [tail comp x0 la_mu la_var]  one row per installed unit
%   P : K x 2  [beta threshold]             per component type
%   usage : T x 2  [usage hours/month, sd of log monthly usage] per tail.
%         The volatility column is ESTIMATED upstream from each tail's own
%         monthly usage history. An earlier version hardcoded 0.15 here --
%         which was the generator's hidden truth constant, copied into the
%         estimator. Nothing the pipeline is asked to infer may be typed in
%         from the simulator's source (results/failures.md).
%   H : horizon in months;  R : replications
%   tat : struct with fields mu, sigma (lognormal repair turnaround, months)
%         and p_spare (probability a serviceable spare is on the shelf; if
%         not, an additional tat.backorder months of downtime)
%
%   Each replication draws alpha from the unit's posterior (epistemic) and
%   then a fresh gamma-process path (aleatory), so the spread of the output
%   reflects both sources.  A tail is mission-capable in a month when none of
%   its components are awaiting repair.
%
%   STATED LIMITATION: severity and usage are drawn INDEPENDENTLY per unit,
%   but a real tail's components share environment and duty, so tail-down
%   events are positively correlated and this MC understates the spread of
%   fleet availability (and overstates P(any component down)). Modelling the
%   shared factor needs a hierarchical tail effect the current fit does not
%   separate from per-unit severity.
%
%   out.avail      : H x 3  [mean p05 p95] fleet availability per month
%   out.fail_comp  : K x H x 3 expected failures per component per month [mean p05 p95]
%   out.fail_total : H x 3
%   out.n_units, out.R

  % 'state' seeds work in both Octave and MATLAB; gamma draws go through
  % gamma_sample (randn/rand only), so these two streams are the only
  % randomness there is -- randg, whose separate state once ran unseeded
  % here and which base MATLAB lacks, is no longer used anywhere.
  rand('state', seed); randn('state', seed);
  usage_rate = usage(:, 1); log_sd = usage(:, 2);
  if any(~isfinite(usage_rate)) || any(~isfinite(log_sd) | log_sd < 0 | log_sd > 2)
    error('avail:bad_usage', 'non-finite usage rate or implausible log-usage sd');
  end
  M = size(U, 1); K = size(P, 1); T = numel(usage_rate);
  tail = U(:,1); comp = U(:,2);
  beta = P(comp, 1); L = P(comp, 2);
  avail = zeros(R, H); fail_c = zeros(R, K, H);
  for rep = 1:R
    alpha = exp(U(:,4) + sqrt(U(:,5)) .* randn(M, 1));
    X = U(:,3);

    % A NaN damage state is SILENTLY IMMORTAL here: X >= L is false for
    % NaN, so the unit never fails, never enters the failure count, and
    % counts as mission-capable for the whole horizon. Two routes in -- a
    % NaN x0 from upstream, and a zero shape (alpha underflow) reaching the
    % gamma sampler. Both are guarded rather than left to produce a
    % plausible availability curve.
    if any(~isfinite(X)) || any(~isfinite(alpha)) || any(alpha <= 0)
      error('avail:nonfinite', ...
            ['Non-finite or non-positive state entering the availability ' ...
             'MC (%d bad damage states, %d bad severities).'], ...
            sum(~isfinite(X)), sum(~isfinite(alpha) | alpha <= 0));
    end
    down_until = -ones(M, 1);
    for m = 1:H
      du = usage_rate(tail) .* exp(log_sd(tail) .* randn(M, 1));
      active = down_until < m;
      shp = alpha .* du;
      dX = zeros(M, 1);
      % a zero/negative shape would be NaN territory; gamma_sample refuses
      % rather than returning one, and one NaN here would make that unit
      % immortal for the rest of the horizon.
      draw = active & shp > 0;
      dX(draw) = gamma_sample(shp(draw)) .* beta(draw);
      X = X + dX;
      if any(~isfinite(X))
        error('avail:nonfinite_path', 'Non-finite damage state after month %d.', m);
      end
      f = active & X >= L;
      if any(f)
        nf = nnz(f);
        dt = exp(tat.mu + tat.sigma * randn(nf, 1));
        nospare = rand(nf, 1) > tat.p_spare;
        dt(nospare) = dt(nospare) + tat.backorder;
        down_until(f) = m + ceil(dt);
        X(f) = 0;
        fail_c(rep, :, m) = accumarray(comp(f), 1, [K 1])';
      end
      tail_down = accumarray(tail, double(down_until >= m), [T 1]) > 0;
      avail(rep, m) = 1 - mean(tail_down);
    end
  end
  out.avail = [mean(avail, 1)' q_(avail, 0.05)' q_(avail, 0.95)'];
  ft = squeeze(sum(fail_c, 2));
  if R == 1, ft = ft'; end
  out.fail_total = [mean(ft, 1)' q_(ft, 0.05)' q_(ft, 0.95)'];
  fc = zeros(K, H, 3);
  for k = 1:K
    a = squeeze(fail_c(:, k, :));
    if R == 1, a = a'; end
    fc(k, :, :) = reshape([mean(a, 1)' q_(a, 0.05)' q_(a, 0.95)'], [1 H 3]);
  end
  out.fail_comp = fc;
  out.n_units = M; out.R = R;
end

function q = q_(A, p)
  A = sort(A, 1); n = size(A, 1);
  q = A(max(1, min(n, round(p * n))), :);
end
