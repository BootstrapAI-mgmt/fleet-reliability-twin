function run_tests()
%RUN_TESTS  Numerics tests: parameter recovery on simulated data with known truth.
%   octave --no-gui -q --eval "addpath('matlab'); addpath('matlab/tests'); run_tests"
  rand('seed', 3); randn('seed', 3);
  n_fail = 0;
  n_fail = n_fail + check('gauss-hermite integrates lognormal mean', test_gh());
  n_fail = n_fail + check('rul CDF equals closed form for known alpha', test_rul_cdf());
  n_fail = n_fail + check('rul quantiles invert the CDF', test_rul_quantiles());
  n_fail = n_fail + check('gamma-process fit recovers beta, sigma, alpha0, tau', test_fit());
  n_fail = n_fail + check('detect: stuck and bias are isolated, clean fleet is quiet', test_detect());
  n_fail = n_fail + check('availability MC bounded and monotone in failures', test_avail());
  if n_fail > 0, error('%d test(s) failed', n_fail); end
  printf('all MATLAB tests passed\n');
end

function f = check(name, ok)
  if ok, printf('  PASS  %s\n', name); f = 0; else printf('  FAIL  %s\n', name); f = 1; end
end

function ok = test_gh()
  % 16-point Gauss-Hermite marginalisation over log alpha ~ N(mu, v) must
  % agree with brute-force numerical integration on a fine grid
  mu = log(0.02); v = 0.3^2; beta = 0.5; L = 6; x0 = 1; u = 40; t = 10;
  F = rul_cdf(x0, mu, v, beta, L, u, t);
  z = linspace(-6, 6, 20001); w = exp(-z.^2 / 2) / sqrt(2 * pi) * (z(2) - z(1));
  a = exp(mu + sqrt(v) * z);
  Fnum = sum(w .* (1 - gammainc((L - x0) / beta, a * u * t)));
  ok = abs(F - Fnum) < 1e-5;
end

function ok = test_rul_cdf()
  alpha = 0.02; beta = 0.5; L = 6; x0 = 1; u = 40; t = 10;
  F = rul_cdf(x0, log(alpha), 1e-12, beta, L, u, t);
  Fexact = 1 - gammainc((L - x0) / beta, alpha * u * t);
  ok = abs(F - Fexact) < 1e-6;
end

function ok = test_rul_quantiles()
  x0 = [1; 3]; la = log([0.02; 0.03]); v = [0.05; 0.2]; beta = [0.5; 0.5]; L = [6; 6]; u = [40; 40];
  [tq, pf] = rul_quantiles(x0, la, v, beta, L, u, [0.05 0.5 0.95], 24);
  Fq = zeros(2, 3);
  for j = 1:3, Fq(:, j) = rul_cdf(x0, la, v, beta, L, u, tq(:, j)); end
  ok = all(abs(Fq - [0.05 0.5 0.95; 0.05 0.5 0.95])(:) < 1e-4) && all(tq(:,1) < tq(:,2)) ...
       && all(tq(:,2) < tq(:,3)) && all(pf >= 0 & pf <= 1);
end

function D = synth(n_chan, n_months, alpha0, beta, sigma, tau, K, k)
  % one component type, channel index k, gamma-process damage with noise
  rows = [];
  for c = 1:n_chan
    a = alpha0 * exp(tau * randn); X = 0; serial = c * 1000;
    for m = 0:n_months - 1
      du = 40 * exp(0.1 * randn);
      X = X + randg(a * du) * beta;
      if X > 8, X = 0; serial = serial + 1; end
      rows(end + 1, :) = [c, k, serial, m, du, X + sigma * randn];
    end
  end
  D = rows;
end

function ok = test_fit()
  D = synth(300, 48, 0.03, 0.2, 0.1, 0.4, 1, 1);
  fit = fit_gamma_process(D, 1, []);
  c = fit.comp(1);
  ok = abs(c.beta / 0.2 - 1) < 0.15 && abs(c.sigma / 0.1 - 1) < 0.3 && ...
       abs(exp(c.mu) / 0.03 - 1) < 0.15 && abs(c.tau - 0.4) < 0.12 && ...
       all(fit.unit(:, 7) >= 0 & fit.unit(:, 7) <= 1);
  if ~ok, printf('    beta %.3f sigma %.3f alpha0 %.4f tau %.3f\n', c.beta, c.sigma, exp(c.mu), c.tau); end
end

function ok = test_detect()
  K = 1; n = 48;
  D = synth(200, n, 0.03, 0.2, 0.1, 0.4, K, 1);
  % channel 1: stuck from month 30; channel 2: bias step +3 at month 20
  i1 = D(:,1) == 1 & D(:,4) >= 30; D(i1, 6) = D(find(i1, 1), 6);
  i2 = D(:,1) == 2 & D(:,4) >= 20; D(i2, 6) = D(i2, 6) + 3;
  fit = fit_gamma_process(D, K, []);
  flags = detect_faults(D, fit, K, n);
  stuck_ok = flags(1, 4) == 3 && abs(flags(1, 5) - 30) <= 3;
  bias_ok  = flags(2, 4) == 1;
  clean = flags(3:end, 4);
  quiet = mean(clean > 0) < 0.03;
  ok = stuck_ok && bias_ok && quiet;
  if ~ok, printf('    ch1 cls %d onset %g; ch2 cls %d; clean flagged %.3f\n', flags(1,4), flags(1,5), flags(2,4), mean(clean > 0)); end
end

function ok = test_avail()
  U = [1 1 5.9 log(0.03) 0.01; 1 2 0.1 log(0.03) 0.01; 2 1 0.1 log(0.03) 0.01];
  P = [0.2 6; 0.2 6];
  tat = struct('mu', 0.3, 'sigma', 0.3, 'p_spare', 0.9, 'backorder', 2);
  out = availability_mc(U, P, [40; 40], 6, 50, tat, 1);
  a = out.avail(:, 1);
  ok = all(a >= 0 & a <= 1) && all(out.avail(:,2) <= out.avail(:,3)) && out.fail_total(1, 1) > 0.5 ...
       && a(1) < 1;   % unit near threshold fails in month 1 and takes tail 1 down
end
