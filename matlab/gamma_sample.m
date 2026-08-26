function g = gamma_sample(shape)
%GAMMA_SAMPLE  Gamma(shape, 1) draws using only randn/rand.
%
%   Replaces Octave's randg, for two reasons. Portability: randg does not
%   exist in base MATLAB, and the toolbox variant cannot be seeded with the
%   legacy 'seed' form -- so every advertised "runs on MATLAB unchanged"
%   path that touched randg was false. Determinism: randg carries its own
%   generator state, which is exactly how this suite once ran unseeded
%   behind seeded rand/randn (results/failures.md). Drawing through
%   randn/rand means the two streams that are seeded are the only streams
%   there are.
%
%   Marsaglia & Tsang (2000), "A simple method for generating gamma
%   variables": squeeze/rejection for shape >= 1, and the boost
%   G(a) = G(a+1) * U^(1/a) for shape < 1. Vectorised; acceptance is
%   ~96% per round so the loop runs 1-2 iterations in practice.

  orig = size(shape);
  shape = shape(:);
  if any(~isfinite(shape) | shape <= 0)
    error('gamma_sample:bad_shape', '%d non-finite or non-positive shape(s)', ...
          sum(~isfinite(shape) | shape <= 0));
  end
  small = shape < 1;
  a = shape; a(small) = a(small) + 1;
  d = a - 1/3; c = 1 ./ sqrt(9 * d);
  g = zeros(numel(a), 1);
  todo = true(numel(a), 1);
  while any(todo)
    idx = find(todo);
    x = randn(numel(idx), 1);
    v = (1 + c(idx) .* x) .^ 3;
    u = rand(numel(idx), 1);
    ok = (v > 0) & (log(u) < 0.5 * x .^ 2 + d(idx) .* (1 - v + log(max(v, realmin))));
    g(idx(ok)) = d(idx(ok)) .* v(ok);
    todo(idx(ok)) = false;
  end
  if any(small)
    g(small) = g(small) .* rand(nnz(small), 1) .^ (1 ./ shape(small));
  end
  g = reshape(g, orig);
end
