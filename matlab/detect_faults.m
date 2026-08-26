function [flags, calib] = detect_faults(D, fit, K, n_months)
%DETECT_FAULTS  Anomaly detection and fault isolation on sensor channels.
%
%   [flags, calib] = DETECT_FAULTS(D, fit, K, n_months)
%
%   A channel is one (tail, component) sensor; it sees a sequence of component
%   instances (serials).  Each reading increment is standardized against the
%   unit's fitted gamma-process prediction.  Five structured residuals give
%   a signature that isolates the fault class:
%
%     E1 spike     : single increment with gamma-tail p < p_spike
%     E2 persist   : growing-window scan from candidate onsets; exact gamma
%                    tail p-value, Bonferroni over onsets (persistent rate change)
%     E3 stuck     : >= 3 consecutive exactly-repeated readings
%     E4 dropout   : >= 40% of the trailing 12 months missing (>= 5 missing)
%     E5 install   : first reading of a freshly installed unit far from zero
%
%   Signature table (rows = fault class, columns = E1 E2 E3 E4 E5):
%     bias_step    1 0 0 0 1      (one jump, then normal increments; shows at
%                                  next install as a non-zero offset)
%     scale_error  1 1 0 0 0      (jump at onset, then inflated increments)
%     accelerated  0 1 0 0 0      (no jump; increments inflated - real damage)
%     stuck        0 0 1 0 *
%     dropout      0 0 0 1 *
%
%   Where the statistics live and where the judgement lives, stated plainly:
%   E1 and E2 are p-values against the fitted process model; E1 and E5 also
%   require a physically large magnitude (4 sigma / 5 sigma floors) so a
%   statistically-surprising-but-tiny wobble cannot flag; E3 and E4 are
%   structural rules (exact repeats; missing fraction), not tests. The
%   floors and structural constants are CHOSEN, and the honest defence of a
%   chosen constant is measurement: the achieved false-alarm rate and power
%   are scored against hidden truth in verify.py rather than assumed.
%
%   flags : C x 10 matrix, one row per channel (tail-1)*K+comp:
%      [chan tail comp class onset_month confidence evidence_bits n_obs watch persist_p]
%      class: 0 none, 1 bias_step, 2 scale_error, 3 stuck, 4 dropout, 5 accelerated
%   calib : struct with the calibrated thresholds (reported for audit)

  p_spike = 2e-5;
  D = sortrows(D, [1 2 4]);                       % tail, comp, month
  tail = D(:,1); comp = D(:,2); serial = D(:,3); month = D(:,4);
  usage = D(:,5); r = D(:,6);
  chan = (tail - 1) * K + comp;
  C = max(chan);

  % channel posterior alpha lookup
  u = fit.unit;
  [~, loc] = ismember(chan, u(:,1));
  alpha = NaN(size(serial)); alpha(loc > 0) = u(loc(loc > 0), 4);
  beta = arrayfun(@(k) fit.comp(k).beta, comp);
  sig  = arrayfun(@(k) fit.comp(k).sigma, comp);

  % ---- per-row increment statistics -----------------------------------
  N = numel(r);
  same_serial = [false; serial(2:end) == serial(1:end-1)];
  same_chan   = [false; chan(2:end) == chan(1:end-1)];
  gapm = [0; month(2:end) - month(1:end-1)];
  du = usage .* gapm;
  d  = [0; r(2:end) - r(1:end-1)];
  inc = same_serial & du > 0 & ~isnan(alpha);
  mean_d = alpha .* beta .* du;
  var_d  = alpha .* beta.^2 .* du + 2 * sig.^2;
  z = NaN(N, 1); z(inc) = (d(inc) - mean_d(inc)) ./ sqrt(var_d(inc));
  % upper-tail p-value of the raw increment under a moment-matched gamma
  % that includes sensor noise (mean alpha*beta*du, var alpha*beta^2*du+2sigma^2).
  % Ignoring the noise term produced >100 false spikes on the smooth,
  % low-beta components where noise dominates (results/failures.md).
  pval = ones(N, 1);
  shp1 = mean_d(inc).^2 ./ var_d(inc); scl1 = var_d(inc) ./ mean_d(inc);
  pval(inc) = 1 - gammainc(max(d(inc), 0) ./ scl1, shp1);
  spike = inc & pval < p_spike & d > 4 * sig;    % require a physically large jump too

  % install offset: first reading of a serial that is not the channel's first
  new_install = ~same_serial & same_chan;
  inst_off = new_install & abs(r) > 5 * sig;

  % ---- E2: persistent rate change by growing-window change-point scan ----
  % For each channel and candidate onset j (every 3rd increment) the rate
  % BEFORE j is estimated from the channel's own pre-onset increments blended
  % with the fleet prior (posterior on log alpha).  The post-onset sum of
  % within-serial increments S is then tested against that pre-onset rate:
  %   S ~ Gamma(alpha_pre * Sdu_post, beta) + noise,
  % with the uncertainty of alpha_pre folded into the variance, and a
  % moment-matched gamma giving the upper-tail p-value.  Noise variance is
  % 2 sigma^2 per contiguous serial segment (increments telescope).  The
  % channel statistic is the minimum over onsets, Bonferroni corrected.
  % Testing against the pre-onset rate rather than the fitted channel rate
  % matters: a long-running acceleration is otherwise absorbed into the
  % channel's own alpha and becomes invisible (results/failures.md).
  p_persist = 1e-3;            % corrected channel-level false-alarm rate
  p_watch   = 1e-2;            % uncorrected "watch list" tier for review
  pers_p = ones(C, 1); pers_on = 1e9 * ones(C, 1); watch = false(C, 1);
  cd_ = [0; cumsum(d .* inc)]; cdu = [0; cumsum(du .* inc)]; cn = [0; cumsum(double(inc))];
  seg = cumsum(double(new_install));
  mu_k  = arrayfun(@(k) fit.comp(k).mu,  1:K);
  tau_k = arrayfun(@(k) fit.comp(k).tau, 1:K);
  [cu, first_row] = unique(chan, 'first');
  last_row = [first_row(2:end) - 1; N];
  for q = 1:numel(cu)
    f = first_row(q); l = last_row(q);
    if l - f < 12, continue; end
    k = comp(f); bk = beta(f); s2 = sig(f)^2; mk = mu_k(k); t2 = tau_k(k)^2;
    js = (f:3:(l - 6))';
    % pre-onset data (rows f..js-1) -> posterior on log alpha_pre
    Sd_pre = cd_(js) - cd_(f); Su_pre = cdu(js) - cdu(f); n_pre = cn(js) - cn(f);
    r_pre = max(Sd_pre ./ max(Su_pre, eps), 1e-6);
    la_pre = log(r_pre / bk);
    % Segments, not increments -- the rule this file states at the top and
    % applies correctly to the post-onset window below. Sd_pre telescopes
    % within each serial, so it carries 2*sigma^2 once per segment.
    jp = max(js - 1, f);
    nseg_pre = seg(jp) - seg(f) + 1;
    v_pre = max((r_pre .* bk .* Su_pre + 2 * s2 .* nseg_pre) ./ max(Su_pre, eps).^2 ./ r_pre.^2, 1e-4);
    v_pre(n_pre < 3) = 1e6;                      % no usable history -> prior only
    pv_ = 1 ./ (1 / t2 + 1 ./ v_pre);
    pm_ = pv_ .* (mk / t2 + la_pre ./ v_pre);
    a_pre = exp(pm_ + pv_ / 2);
    % post-onset window (rows js..l)
    S  = cd_(l + 1) - cd_(js);
    Su = cdu(l + 1) - cdu(js);
    m  = a_pre .* bk .* Su;
    v  = a_pre .* bk^2 .* Su + m.^2 .* (exp(pv_) - 1) + 2 * s2 .* (seg(l) - seg(js) + 1);
    okj = m > 0 & v > 0 & Su > 0;
    if ~any(okj), continue; end
    shp = m(okj).^2 ./ v(okj); scl = v(okj) ./ m(okj);
    pv = 1 - gammainc(max(S(okj), 0) ./ scl, shp);
    [pmin, im] = min(pv);
    jj = js(okj); on = month(jj(im));
    pc = min(1, pmin * nnz(okj));
    pers_p(cu(q)) = pc; pers_on(cu(q)) = on;
    watch(cu(q)) = pmin < p_watch;
  end
  persist_c = pers_p < p_persist;
  calib = struct('p_spike', p_spike, 'p_persist', p_persist, 'p_watch', p_watch, ...
                 'n_channels', C);

  % exact repeats (stuck)
  rep = same_chan & abs(d) < 1e-9;
  % run length of consecutive repeats (vectorised)
  run = cumsum(rep) - cummax(cumsum(rep) .* ~rep);
  stuck_row = run >= 2;                           % 3 identical readings

  % ---- dropout: trailing-12-month coverage per channel ----------------
  last_m = n_months - 1;
  trailing = month > last_m - 12;
  n_tr = accumarray(chan, double(trailing), [C 1]);
  first_seen = accumarray(chan, month, [C 1], @min, NaN);
  span = min(12, last_m - first_seen + 1);
  missing = span - n_tr;
  dropout_c = isfinite(span) & missing >= 5 & missing ./ max(span,1) >= 0.4;

  % ---- per-channel evidence aggregation -------------------------------
  first_or_nan = @(m) accumarray(chan, month .* m + 1e9 * ~m, [C 1], @min, 1e9);
  t_spike = first_or_nan(spike);
  t_pers  = pers_on;
  t_stuck = first_or_nan(stuck_row);
  t_inst  = first_or_nan(inst_off);
  has = @(t) t < 1e9;
  E1 = has(t_spike); E2 = persist_c; E3 = has(t_stuck); E4 = dropout_c; E5 = has(t_inst);
  n_obs = accumarray(chan, 1, [C 1]);

  flags = zeros(C, 10);
  flags(:,1) = (1:C)'; flags(:,2) = floor(((1:C)' - 1) / K) + 1;
  flags(:,3) = mod((1:C)' - 1, K) + 1; flags(:,8) = n_obs;
  for c = 1:C
    cls = 0; onset = NaN; conf = 0;
    bits = E1(c) + 2*E2(c) + 4*E3(c) + 8*E4(c) + 16*E5(c);
    if E4(c)
      cls = 4; onset = last_m - 12 + 1; conf = min(1, missing(c) / 12);
    elseif E3(c)
      cls = 3; onset = t_stuck(c) - 2; conf = 1;
    elseif E2(c)
      if E1(c) && t_spike(c) <= t_pers(c)
        cls = 2; onset = t_spike(c);
      else
        cls = 5; onset = t_pers(c);
      end
      conf = min(1, -log10(max(pers_p(c), 1e-12)) / 6);
    elseif E1(c) || E5(c)
      cls = 1; onset = min(t_spike(c), t_inst(c)); conf = 0.7 + 0.3 * (E1(c) && E5(c));
    end
    flags(c, 4) = cls; flags(c, 5) = onset; flags(c, 6) = conf; flags(c, 7) = bits;
    flags(c, 9) = watch(c) && cls == 0;          % watch-list tier, no hard flag
    flags(c, 10) = pers_p(c);
  end
end

