function [y, fval] = dualInference(w, featureMap, kappa, S, ell)

% performs inference or loss-augmented inference in the dual

if ~exist('ell', 'var')
    ell = zeros(size(featureMap,2),1);
end

if kappa == 0
    [y, fval] = inference(w, featureMap, kappa, S, ell);
    return;
end

Ftwlk1 = (featureMap'*w + ell)/kappa - 1;

fun = @(y, varargin) obj(y, Ftwlk1, kappa, S, varargin);

options.Display = 'off';
% options.outputFcn = @inferenceStat;
options.Method = 'lbfgs';
options.GradObj = 'on';
options.MaxFunEvals = 8000;
options.MaxIter = 8000;
options.LS_type = 0;
options.LS_interp = 0;
% options.progTol = 1e-6;
% options.optTol = 1e-3;

% lambda = zeros(size(S.Aeq,1),1);
lambda = initLambda(featureMap'*w + ell, kappa, S.Aeq);

[lambda, fval] = minFunc(fun,lambda,options, fun);

y = exp((featureMap' * w + ell + S.Aeq'*lambda)/kappa - 1);


function [f,g] = obj(lambda, Ftwlk1 , kappa, S, varargin)

%y = exp((Ftw + ell + S.Aeq'*lambda)/kappa - 1);
y = exp(Ftwlk1 + S.Aeq'*lambda / kappa);

f = kappa * sum(y) - S.beq'*lambda;

g = S.Aeq * y - S.beq;


function lambda = initLambda(Ftwl, kappa, A)

c = size(A,1);
options.MSK_IPAR_INTPNT_NUM_THREADS = 4;
options.Display = 'notify';
lambda = quadprog(speye(c), zeros(c,1), A'/kappa, -Ftwl/kappa, [], [], [], [], [], options);
