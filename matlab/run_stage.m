function run_stage(stage, work)
%RUN_STAGE  File-based entry point for the Python orchestrator.
%   run_stage('fit'|'detect'|'rul'|'avail', workdir)
%   Every stage reads only from `work` and writes only to `work`.  Nothing is
%   held in MATLAB state between stages, so any stage can be re-run in
%   isolation from its checkpoint.

  cfg = jsondecode(fileread(fullfile(work, 'config.json')));
  K = cfg.K;
  switch stage
    case 'fit'
      D = dlmread(fullfile(work, 'clean.csv'), ',', 1, 0);
      exf = fullfile(work, 'exclude.csv');
      ex = [];
      if exist(exf, 'file')
        ex = logical(dlmread(exf));
      end
      fit = fit_gamma_process(D, K, ex);
      % unit table: chan comp tail alpha la_mu la_var shrink n_incr
      dlmwrite(fullfile(work, 'fit_units.csv'), fit.unit, 'precision', '%.8g');
      fid = fopen(fullfile(work, 'fit_comp.json'), 'w');
      fprintf(fid, '%s', jsonencode(fit.comp)); fclose(fid);
    case 'detect'
      D = dlmread(fullfile(work, 'clean.csv'), ',', 1, 0);
      fit = load_fit(work, K);
      [flags, calib] = detect_faults(D, fit, K, cfg.n_months);
      dlmwrite(fullfile(work, 'flags.csv'), flags, 'precision', '%.6g');
      fid = fopen(fullfile(work, 'detect_calib.json'), 'w');
      fprintf(fid, '%s', jsonencode(calib)); fclose(fid);
    case 'rul'
      % columns: unit_id x0 la_mu la_var beta L usage_per_month
      A = dlmread(fullfile(work, 'rul_in.csv'), ',', 1, 0);
      probs = [0.05 0.5 0.95];
      H = cfg.forecast_months;
      [tq, pf] = rul_quantiles(A(:,2), A(:,3), A(:,4), A(:,5), A(:,6), A(:,7), probs, H);
      % failure CDF at each month 1..H (for calibration and expected-count checks)
      Fm = zeros(size(A,1), H);
      for m = 1:H
        Fm(:, m) = rul_cdf(A(:,2), A(:,3), A(:,4), A(:,5), A(:,6), A(:,7), m);
      end
      dlmwrite(fullfile(work, 'rul_out.csv'), [A(:,1) tq pf Fm], 'precision', '%.6g');
    case 'avail'
      U = dlmread(fullfile(work, 'avail_units.csv'), ',', 1, 0);
      P = dlmread(fullfile(work, 'comp_params.csv'), ',', 1, 0);
      ur = dlmread(fullfile(work, 'usage.csv'), ',', 1, 0);
      tat = cfg.tat;
      % columns 2:3 are [usage/month, sd of log monthly usage], both
      % estimated upstream from the tail's own history
      out = availability_mc(U, P, ur(:, 2:3), cfg.forecast_months, cfg.mc_reps, tat, cfg.seed);
      fid = fopen(fullfile(work, 'avail.json'), 'w');
      fprintf(fid, '%s', jsonencode(out)); fclose(fid);
    otherwise
      error('run_stage:unknown', 'unknown stage %s', stage);
  end
end

function fit = load_fit(work, K)
  fit.unit = dlmread(fullfile(work, 'fit_units.csv'));
  c = jsondecode(fileread(fullfile(work, 'fit_comp.json')));
  for k = 1:K
    fit.comp(k) = c(k);
  end
  fit.K = K;
end
