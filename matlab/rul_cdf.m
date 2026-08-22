function F = rul_cdf(x0, la_mu, la_var, beta, L, u, t)
%RUL_CDF  P(failure within t months), same model as rul_quantiles, no search.
  n = numel(x0);
  [gh_x, gh_w] = gauss_hermite(16);
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
