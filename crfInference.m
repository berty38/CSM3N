function y = crfInference(w, featureMap, singletons, S)


% performs crf inference in the dual

fun = @(lambda, varargin) obj(lambda, w, featureMap, S, singletons, varargin);

[d,m] = size(featureMap);
s = ones(m,1);
s(singletons+1:end) = 1;

options.Display = 'off';
options.outputFcn = @inferenceStat;
options.Method = 'lbfgs';
options.GradObj = 'on';
options.MaxFunEvals = 8000;
options.MaxIter = 8000;
options.LS_type = 0;
options.LS_interp = 0;
% options.progTol = 1e-6;
% options.optTol = 1e-3;

% lambda = zeros(size(S.Aeq,1),1);
lambda = initLambda(featureMap'*w, S.Aeq);

[lambda, fval] = minFunc(fun, lambda, options, fun);

y = exp(s .* (featureMap'*w + S.Aeq'*lambda) - 1);


function [f,g] = obj(lambda, w, F, S, singletons, varargin)

[d,m] = size(F);

s = ones(m,1);
s(singletons+1:end) = 1;

y = exp(s .* (F'*w + S.Aeq'*lambda) - 1);

f =  sum(y) - S.beq'*lambda;

if nargout == 2
    g = S.Aeq * (s .* y) - S.beq;
end




function lambda = initLambda(Ftw, A)

c = size(A,1);
options.MSK_IPAR_INTPNT_NUM_THREADS = 4;
options.Display = 'notify';
lambda = quadprog(speye(c), zeros(c,1), A', -Ftw, [], [], [], [], [], options);
