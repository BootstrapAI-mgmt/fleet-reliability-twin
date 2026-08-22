function [tq, pfail] = rul_quantiles(x0, la_mu, la_var, beta, L, u, probs, horizon)
%RUL_QUANTILES  Remaining-useful-life quantiles for gamma-process deterioration.
%
%   [tq, pfail] = RUL_QUANTILES(x0, la_mu, la_var, beta, L, u, probs, horizon)
%
%   For a gamma process with shape alpha*du and scale beta, the first-passage
%   time of threshold L starting from damage x0 has the closed-form CDF
%       F(t) = P(X(t) >= L - x0) = 1 - gammainc((L-x0)/beta, alpha*u*t)
%   where u is usage per month.  Uncertainty in alpha (posterior on log alpha
%   ~ N(la_mu, la_var)) is integrated out by Gauss-Hermite quadrature, so the
%   returned quantiles are of the posterior-predictive RUL.  No Monte Carlo.
%
%   All inputs except probs/horizon are column vectors (one row per unit).
%   tq    : numel(x0) x numel(probs) quantiles in months (Inf if beyond horizon)
%   pfail : probability of failure within `horizon` months
%
%   Base MATLAB / Octave (gammainc is core).

  n = numel(x0);
  [gh_x, gh_w] = gauss_hermite(16);          % nodes/weights for N(0,1)
  alphas = exp(la_mu + sqrt(2 * la_var) * gh_x');   % n x 16
  w = (gh_w / sqrt(pi))';                           % 1 x 16
  z = max(L - x0, 1e-9) ./ beta;                    % gamma-scaled remaining margin
  Z = z(:, ones(1, numel(gh_w)));                  % explicit expansion (gammainc does not broadcast)
  W = w(ones(n, 1), :);

  F = @(t) 1 - sum(W .* gammainc(Z, alphas .* ((u .* t) * ones(1, numel(gh_w)))), 2);  % t: n x 1
  % gammainc(x, a) is the regularized lower incomplete gamma P(a, x).
  pfail = F(horizon * ones(n, 1));

  tq = zeros(n, numel(probs));
  for j = 1:numel(probs)
    p = probs(j);
    lo = zeros(n, 1); hi = 4 * horizon * ones(n, 1);
    beyond = F(hi) < p;                      % quantile is past search bound
    for it = 1:32
      mid = 0.5 * (lo + hi);
      fm = F(mid);
      up = fm < p;
      lo(up) = mid(up); hi(~up) = mid(~up);
    end
    t = 0.5 * (lo + hi);
    t(beyond) = Inf;
    tq(:, j) = t;
  end
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
