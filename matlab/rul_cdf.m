function F = rul_cdf(x0, la_mu, la_var, beta, L, u, t)
%RUL_CDF  P(failure within t months), same model as rul_quantiles, no search.
  n = numel(x0);
  if any(~isfinite(x0)) || any(~isfinite(la_mu)) || any(~isfinite(la_var)) || ...
     any(~isfinite(beta)) || any(~isfinite(L)) || any(~isfinite(u))
    error('rul:nonfinite', ...
          ['Non-finite input to the RUL model. This is guarded rather than ' ...
           'clamped because MAX and MIN silently DROP NaN: max(L-x0, 1e-9) ' ...
           'with a NaN x0 returns 1e-9, which the model reads as "already ' ...
           'at threshold, fails within days", and the downstream 0<=p<=1 ' ...
           'gate accepts it. A missing damage state must stop the run, not ' ...
           'become maximum alarm.']);
  end

  % 64 nodes, not 16. Against a 4001-point reference the 16-node result was
  % off by up to 0.028 in per-unit P(fail) -- larger than the reported
  % calibration error of the whole model -- with the worst cases being the
  % channels whose posterior spread is widest, where the integrand is a
  % sharp sigmoid in the quadrature variable. Convergence sets in near 64.
  % The fleet total was fine either way because the errors cancel; the
  % per-channel number, which is the one an operator acts on, was not.
  [gh_x, gh_w] = gauss_hermite(64);
  alphas = exp(la_mu + sqrt(2 * la_var) * gh_x');
  w = (gh_w / sqrt(pi))';
  z = max(L - x0, 1e-9) ./ beta;
  Z = z(:, ones(1, numel(gh_w))); W = w(ones(n, 1), :);
  if isscalar(t), t = t * ones(n, 1); end
  F = 1 - sum(W .* gammainc(Z, alphas .* ((u .* t) * ones(1, numel(gh_w)))), 2);
end

function [x, w] = gauss_hermite(m)
  % Golub-Welsch for physicists' Hermite polynomials: integrates exp(-x^2) f(x)
  i = (1:m-1)';
  b = sqrt(i / 2);
  J = diag(b, 1) + diag(b, -1);
  [V, Dm] = eig(J);
  x = diag(Dm);
  w = sqrt(pi) * V(1, :)'.^2;
  [x, o] = sort(x); w = w(o);
end
