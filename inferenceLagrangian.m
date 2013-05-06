function [f,g] = inferenceLagrangian(y, lambda, gamma, kappa, featureMap, w, ell, S)

% objective value and gradient for full inference Lagrangian. Used for
% derivations; probably not useful for actual algorithms

if kappa > 0 && any(y<=0)
    f = inf;
end
f = (featureMap'*w + ell + gamma)'*y - kappa * y(y>0)' * log(y(y>0)) + lambda'*(S.Aeq*y - S.beq);
f = -f;

g = featureMap'*w + ell + gamma - kappa * (1 + log(y)) + S.Aeq'*lambda;
g = -g;