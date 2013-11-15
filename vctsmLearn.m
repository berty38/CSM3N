function [w, kappa, y, x] = vctsmLearn(featureMap, labels, scope, S, C, x0)

% Optimizes the VCTSM objective, learning the optimal (w,kappa) 
% and worst violator y, for a given featureMap, label set, MRF structure (S), 
% and slack parameter (C).

func = @(y, varargin) vctsmObj(...
	y, featureMap, labels, scope, S, C, featureMap*labels, varargin);

ell = zeros(size(labels));
ell(scope) = 1 - 2*labels(scope);

[d,m] = size(featureMap);

c = size(S.Aeq,1);

% check for initialization
if ~exist('x0', 'var') || isempty(x0)
    x0 = zeros(d+c+1,1);
end

% optimization options
clear options;
% for minFunc
options.Corr = 200;
options.LS_type = 0;
options.LS_interp = 0;
options.Display = 'off';
% options.outputFcn = @inferenceStat;
options.Method = 'lbfgs';
options.maxIter = 8000;
options.MaxFunEvals = 8000;
options.progTol = 1e-6;
options.optTol = 1e-3;

% run optimization
x = minFunc(func, x0, options, func);

% parse optimization output
w = x(1:d);
kappa = exp(x(d+1));
lambda = x(d+2:end);

% worst violator
y = exp((featureMap'*w + ell + m*S.Aeq'*lambda)/kappa - 1);


