function run_tests()
%RUN_TESTS  Known-parameter recovery tests for the numerics.  octave --eval run_tests
  addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
  rand('seed', 3); randn('seed', 3);
  n_pass = 0; n_fail = 0;
  function check(name, ok)
    if ok, n_pass = n_pass + 1; fprintf('PASS %s\n', name);
    else,  n_fail = n_fail + 1; fprintf('FAIL %s\n', name); end
  end

  % --- synthetic fleet with known alpha/beta/sigma -----------------------
  K = 1; T = 400; months = 48; beta = 0.3; sig = 0.2; a0 = 0.02; tau = 0.4;
  % sigma is weakly identified when 2*sigma^2 << process variance per increment
  % (see README, known limitations); tested here in the identifiable regime.
  alpha = a0 * exp(tau * randn(T, 1)); D = zeros(T * months, 6); r = 0;
  urate = 40 * exp(0.35 * randn(T, 1));            % usage varies by tail, as in a real fleet
  for t = 1:T
    X = 0; serial = t;
    for m = 0:months - 1
      du = urate(t) * exp(0.15 * randn); X = X + randg(alpha(t) * du) * beta;
      r = r + 1; D(r, :) = [t 1 serial m du X + sig * randn];
    end
  end
  fit = fit_gamma_process(D, K, []);
  c = fit.comp(1);
  check(sprintf('beta within 10%% (%.3f vs %.3f)', c.beta, beta), abs(c.beta / beta - 1) < 0.10);
  check('sigma within 20%', abs(c.sigma / sig - 1) < 0.20);
  check('alpha nominal within 10%', abs(exp(c.mu) / a0 - 1) < 0.10);
  check('tau within 25%', abs(c.tau / tau - 1) < 0.25);
  z = (log(alpha(fit.unit(:, 3))) - fit.unit(:, 5)) ./ sqrt(fit.unit(:, 6));
  cov = mean(abs(z) < 1.645);
  check(sprintf('posterior 90%% coverage in [0.85,0.95] (%.3f)', cov), cov > 0.85 && cov < 0.95);
  check('shrinkage in (0,1)', all(fit.unit(:, 7) > 0 & fit.unit(:, 7) < 1));

  % --- RUL CDF against simulation ----------------------------------------
  Lth = 5; u = 40; x0 = 1; a = 0.02; la_var = 0.0;    % known alpha, no posterior spread
  [tq, pf] = rul_quantiles(x0, log(a), la_var, beta, Lth, u, [0.05 0.5 0.95], 24);
  N = 20000; tt = zeros(N, 1);
  for i = 1:N
    X = x0; m = 0;
    while X < Lth, m = m + 1; X = X + randg(a * u) * beta; end
    tt(i) = m;
  end
  F24 = mean(tt <= 24);
  check(sprintf('P(fail<=24) analytic %.3f vs sim %.3f', pf, F24), abs(pf - F24) < 0.02);
  check('median RUL within 1 month of simulated', abs(tq(2) - median(tt)) <= 1);
  check('quantiles monotone', tq(1) <= tq(2) && tq(2) <= tq(3));
  check('rul_cdf consistent with rul_quantiles', abs(rul_cdf(x0, log(a), 0, beta, Lth, u, 24) - pf) < 1e-9);

  % --- detector: stuck channel and clean channel ------------------------
  cfgn = months;
  D2 = D(D(:, 1) <= 50, :);
  D2(D2(:, 1) == 7 & D2(:, 4) >= 20, 6) = D2(D2(:, 1) == 7 & D2(:, 4) == 20, 6);  % stuck from m20
  fit2 = fit_gamma_process(D2, K, []);
  flags = detect_faults(D2, fit2, K, cfgn);
  check('stuck channel isolated as stuck', flags(7, 4) == 3);
  check('stuck onset within 2 months', abs(flags(7, 5) - 20) <= 2);
  check('no hard flags on the 49 clean channels', all(flags([1:6 8:50], 4) == 0));

  % --- availability MC sanity --------------------------------------------
  U = [(1:50)' ones(50, 1) zeros(50, 1) log(a) * ones(50, 1) 0.01 * ones(50, 1)];
  out = availability_mc(U, [beta Lth], 40 * ones(50, 1), 12, 20, ...
                        struct('mu', 0.3, 'sigma', 0.4, 'p_spare', 0.9, 'backorder', 2), 1);
  check('availability in [0,1] and band ordered', all(out.avail(:) >= 0 & out.avail(:) <= 1) && ...
        all(out.avail(:, 2) <= out.avail(:, 1) + 1e-9 & out.avail(:, 1) <= out.avail(:, 3) + 1e-9));
  check('fresh parts: month-1 availability = 1', out.avail(1, 1) == 1);

  fprintf('\n%d passed, %d failed\n', n_pass, n_fail);
  if n_fail > 0, error('tests failed'); end
end
