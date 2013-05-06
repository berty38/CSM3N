function [w, kappa, y] = jointLearnEnt(featureMap, labels, scope, S, C)

% optimizes the joint minimization objective, learning the optimal w,
% kappa, and worst-violator y for a given featureMap, label set, MRF
% structure (S), and slack parameter (C)

% hard code z (for now, 0)
z = 0;

func = @(y) jointObjectiveEnt(y, featureMap, labels, scope, S, C, z);

ell = zeros(size(labels));
ell(scope) = 1 - labels(scope);

[d,m] = size(featureMap);

clear options;
options.verbose = 0;
options.GradObj = 'on';
options.MaxFunEvals = inf;
options.maxIter = 1e9;
options.method = 'lbfgs';
options.correction = 250;

c = size(S.Aeq,1);

lb = -inf(d+c+1, 1);
lb(d+1) = 0;
x0 = zeros(d+c+1,1);
x0(d+1) = 1;

% x = fmincon(func, x0, [], [], [], [], lb, [], [], options);

x = minConf_TMP(func,x0,lb,inf(size(lb)),options);

w = x(1:d);
kappa = x(d+1);

lambda = x(d+2:end);

y = exp((featureMap'*w + ell + S.Aeq'*lambda)/kappa - 1);

