function [w, x] = learnCRF(featureMap, labels, singletons, S, C)

func = @(x) crfObjective(x, featureMap, S, C, singletons, featureMap*labels);

[d,m] = size(featureMap);

s = ones(m,1);
s(1:singletons) = -1;

clear options;
% for minFunc
options.Corr = 200;
options.LS_type = 0;
options.LS_interp = 0;
options.Display = 'off';
options.outputFcn = @inferenceStat;
options.Method = 'lbfgs';
options.maxIter = 8000;
options.MaxFunEvals = 8000;

c = size(S.Aeq,1);

if ~exist('x0', 'var') || isempty(x0)
    x0 = zeros(d + c,1);
end

x = minFunc(func, x0, options);

w = x(1:d);
lambda = x(d+1:end);

y = exp(s .* (featureMap'*w + S.Aeq'*lambda) - 1);



