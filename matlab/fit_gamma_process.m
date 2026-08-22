function fit = fit_gamma_process(D, K, exclude_channel)
%FIT_GAMMA_PROCESS  Hierarchical gamma-process deterioration fit with partial pooling.
%
%   fit = FIT_GAMMA_PROCESS(D, K, exclude_channel)
%
%   D : N x 6 numeric matrix, columns
%       1 tail (1-based index)  2 comp (1-based)  3 serial  4 month
%       5 usage_hours in that month  6 sensor reading of damage X
%   K : number of component types
%   exclude_channel : logical vector over channel id (tail-1)*K+comp, true to
%       drop that channel from the fit (flagged sensor faults).  [] = none.
%
%   Model per component k, per channel c (a channel is one tail x component
%   installation; severity alpha is a property of the installation and its
%   environment, so it is pooled across every serial that has occupied it):
%       dX over usage du ~ Gamma(shape = alpha_c * du, scale = beta_k)
%       reading = X + N(0, sigma_k^2)
%       log alpha_c ~ N(mu_k, tau_k^2)      (fleet prior, estimated EB)
%
%   Fleet-level (beta_k, sigma_k) come from a unit-level moment regression
%   with whole-channel outlier rejection; (mu_k, tau_k) from maximising the
%   marginal likelihood of the per-channel log-rate estimates.  Each channel
%   gets a posterior for log alpha_c that is a precision-weighted blend of
%   its own data and the fleet prior.  The shrinkage factor s_c in [0,1] is
%   the fraction of the posterior BORROWED from the fleet (1 = pure prior).
%
%   Returns struct fit with fields:
%     comp(k).beta, .sigma, .mu, .tau, .n_units, .n_incr, .rejected_units
%     unit : C x 8 matrix [chan comp tail alpha_post_mean logalpha_mu
%                          logalpha_var shrinkage n_incr]
%   Base MATLAB / GNU Octave only (no toolboxes).

  if nargin < 3, exclude_channel = []; end

  % ---- increments within serial, keyed by channel ----------------------
  D = sortrows(D, [3 4]);                      % by serial, then month
  serial = D(:,3); month = D(:,4); usage = D(:,5); r = D(:,6);
  same = [false; serial(2:end) == serial(1:end-1)];
  i2 = find(same); i1 = i2 - 1;
  du = usage(i2) .* (month(i2) - month(i1));
  d  = r(i2) - r(i1);
  comp = D(i2,2); tail = D(i2,1);
  chan = (tail - 1) * K + comp;
  keep = du > 0;
  if ~isempty(exclude_channel)
    ex = exclude_channel(chan); keep = keep & ~ex(:);
  end
  du = du(keep); d = d(keep); comp = comp(keep); tail = tail(keep); chan = chan(keep);

  [useq, ~, uid] = unique(chan);
  M = numel(useq);
  ucomp = accumarray(uid, comp, [M 1], @max);
  utail = accumarray(uid, tail, [M 1], @max);
  n_incr = accumarray(uid, 1, [M 1]);
  Sdu = accumarray(uid, du, [M 1]);
  Sd  = accumarray(uid, d,  [M 1]);
  rate = max(Sd ./ max(Sdu, eps), 1e-6);        % r_c = estimate of alpha_c*beta
  e = d - rate(uid) .* du;
  Ve = accumarray(uid, e.^2, [M 1]);

  unit = zeros(M, 8);
  unit(:,1) = useq; unit(:,2) = ucomp; unit(:,3) = utail; unit(:,8) = n_incr;

  for k = 1:K
    uk = find(ucomp == k);
    if numel(uk) < 10
      fit.comp(k) = struct('beta', NaN, 'sigma', NaN, 'mu', NaN, 'tau', NaN, ...
                           'n_units', numel(uk), 'n_incr', 0, 'rejected_units', 0);
      continue;
    end
    nu = n_incr(uk); Su = Sdu(uk); ru = rate(uk); Vu = Ve(uk);

    % ---- beta, sigma: unit-level moment regression -----------------------
    %   E[sum e^2 * n/(n-1)] = beta * (r_c * Sdu_c) + 2 sigma^2 * n_c
    % with whole-unit outlier rejection (faulted sensors are outlying UNITS;
    % rejecting units does not truncate the gamma tail the way trimming
    % increments does -- see results/failures.md).
    ok_ = nu >= 3;
    y = Vu(ok_) .* nu(ok_) ./ (nu(ok_) - 1);
    A = [ru(ok_) .* Su(ok_), nu(ok_)];
    use = true(size(y));
    for pass = 1:3
      coef = A(use, :) \ y(use);
      res = (y - A * coef) ./ max(sqrt(max(A * coef, eps)), eps);
      res = res / (1.4826 * median(abs(res(use))));
      use = abs(res) < 4;
    end
    beta = max(coef(1), 1e-4);
    sig2 = max(coef(2) / 2, 1e-6);

    % ---- per-unit log alpha and its sampling variance --------------------
    la = log(ru / beta);
    var_r = (ru .* beta .* Su + 2 * sig2 .* nu) ./ Su.^2;
    v = max(var_r ./ ru.^2, 1e-4);              % delta method on log scale

    % ---- mu, tau: marginal likelihood  la_c ~ N(mu, tau^2 + v_c) ---------
    good = nu >= 5 & ru > 1e-5;
    good(ok_) = good(ok_) & use;                 % drop rejected units from prior
    lg = la(good); vg = v(good);
    grid = logspace(-3, 0.5, 120);              % tau^2 candidates
    best = -Inf; mu = 0; tau2 = 0.01;
    for t2 = grid
      w = 1 ./ (t2 + vg);
      m = sum(w .* lg) / sum(w);
      ll = -0.5 * sum(log(t2 + vg)) - 0.5 * sum(w .* (lg - m).^2);
      if ll > best, best = ll; mu = m; tau2 = t2; end
    end

    post_prec = 1 / tau2 + 1 ./ v;
    post_var = 1 ./ post_prec;
    post_mu = post_var .* (mu / tau2 + la ./ v);
    shrink = (1 / tau2) ./ post_prec;
    none = nu < 2;
    post_mu(none) = mu; post_var(none) = tau2; shrink(none) = 1;

    unit(uk, 4) = exp(post_mu + post_var / 2);
    unit(uk, 5) = post_mu;
    unit(uk, 6) = post_var;
    unit(uk, 7) = shrink;
    fit.comp(k) = struct('beta', beta, 'sigma', sqrt(sig2), 'mu', mu, ...
                         'tau', sqrt(tau2), 'n_units', numel(uk), 'n_incr', sum(nu), ...
                         'rejected_units', nnz(~use));
  end
  fit.unit = unit;
  fit.K = K;
end
