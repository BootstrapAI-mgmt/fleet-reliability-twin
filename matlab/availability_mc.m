function out = availability_mc(U, P, usage_rate, H, R, tat, seed)
%AVAILABILITY_MC  Fleet availability and failure-count forecast by Monte Carlo.
%
%   out = AVAILABILITY_MC(U, P, usage_rate, H, R, tat, seed)
%
%   U : M x 5  [tail comp x0 la_mu la_var]  one row per installed unit
%   P : K x 2  [beta threshold]             per component type
%   usage_rate : T x 1 usage hours / month per tail
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
%   out.avail      : H x 3  [mean p05 p95] fleet availability per month
%   out.fail_comp  : K x H x 3 expected failures per component per month [mean p05 p95]
%   out.fail_total : H x 3
%   out.n_units, out.R

  rand('seed', seed); randn('seed', seed);
  M = size(U, 1); K = size(P, 1); T = numel(usage_rate);
  tail = U(:,1); comp = U(:,2);
  beta = P(comp, 1); L = P(comp, 2);
  avail = zeros(R, H); fail_c = zeros(R, K, H);
  for rep = 1:R
    alpha = exp(U(:,4) + sqrt(U(:,5)) .* randn(M, 1));
    X = U(:,3);
    down_until = -ones(M, 1);
    for m = 1:H
      du = usage_rate(tail) .* exp(0.15 * randn(M, 1));
      active = down_until < m;
      shp = alpha .* du;
      dX = zeros(M, 1);
      dX(active) = randg(shp(active)) .* beta(active);
      X = X + dX;
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
