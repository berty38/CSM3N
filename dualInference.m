function [y, fval] = dualInference(w, featureMap, kappa, S, ell)

% performs inference or loss-augmented inference in the dual

if ~exist('ell', 'var')
    ell = zeros(size(featureMap,2),1);
end

fun = @(y) obj(y, exp((featureMap'*w + ell)/kappa - 1), kappa, S);

options.Display = 'off';
options.outputFcn = @inferenceStat;
options.Method = 'lbfgs';
options.GradObj = 'on';
options.MaxFunEvals = 1e12;
options.MaxIter = 1e12;
options.LS_interp = 0;

lambda = zeros(size(S.Aeq,1),1);

[lambda, fval] = minFunc(fun,lambda,options);


y = exp((featureMap' * w + ell + S.Aeq'*lambda)/kappa - 1);


function [f,g] = obj(lambda, eFtwlm1 , kappa, S)

%y = exp((Ftw + ell + S.Aeq'*lambda)/kappa - 1);
y = eFtwlm1 .* exp(S.Aeq'*lambda / kappa);

f = kappa * sum(y)...
    - S.beq'*lambda;

g = S.Aeq * y - S.beq;


