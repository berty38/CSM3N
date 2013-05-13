function [w, kappa, y, x] = jointLearnEnt(featureMap, labels, scope, S, C, x0)

% optimizes the joint minimization objective, learning the optimal w,
% kappa, and worst-violator y for a given featureMap, label set, MRF
% structure (S), and slack parameter (C)

% hard code z (for now, 0)
z = 0;

func = @(y) jointObjectiveEnt(y, featureMap, labels, scope, S, C, z, featureMap*labels);

ell = zeros(size(labels));
ell(scope) = 2*(1 - labels(scope)) - 1;

[d,m] = size(featureMap);

clear options;
% for minConf
options.verbose = 2;
options.GradObj = 'on';
options.MaxFunEvals = inf;
options.maxIter = 8000;
options.method = 'lbfgs';
options.corrections = 200;
options.interp = 0;

% for minFunc
options.Corr = 200;
options.LS_type = 0;
options.LS_interp = 0;
options.Display = 'final';
options.outputFcn = @inferenceStat;
options.Method = 'lbfgs';
% options.progTol = 1e-6;
% options.optTol = 1e-3;

c = size(S.Aeq,1);

lb = -inf(d+c+1, 1);
lb(d+1) = 0;

if ~exist('x0', 'var') || isempty(x0)
    x0 = zeros(d+c+1,1);
    x0(d+1) = 10;
end

% x = minConf_TMP(func,x0,lb,inf(size(lb)),options);
x = minFunc(func, x0, options);

w = x(1:d);
kappa = x(d+1);

lambda = x(d+2:end);

y = exp((featureMap'*w + ell + S.Aeq'*lambda)/kappa - 1);

