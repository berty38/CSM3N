function [y, fval] = dualInference(w, featureMap, kappa, S, ell)

% performs inference or loss-augmented inference in the dual

if ~exist('ell', 'var')
    ell = zeros(size(featureMap,2),1);
end

if kappa == 0
    [y, fval] = inference(w, featureMap, kappa, S, ell);
    return;
end

fun = @(y) obj(y, exp((featureMap'*w + ell)/kappa - 1), kappa, S);

options.Display = 'final';
% options.outputFcn = @inferenceStat;
options.Method = 'lbfgs';
options.GradObj = 'on';
options.MaxFunEvals = 8000;
options.MaxIter = 8000;
options.LS_type = 0;
options.LS_interp = 0;
% options.progTol = 1e-6;
% options.optTol = 1e-3;

lambda = zeros(size(S.Aeq,1),1);

[lambda, fval] = minFunc(fun,lambda,options);


y = exp((featureMap' * w + ell + S.Aeq'*lambda)/kappa - 1);


function [f,g] = obj(lambda, eFtwlm1 , kappa, S)

%y = exp((Ftw + ell + S.Aeq'*lambda)/kappa - 1);
y = eFtwlm1 .* exp(S.Aeq'*lambda / kappa);

f = kappa * sum(y)...
    - S.beq'*lambda;

g = S.Aeq * y - S.beq;


