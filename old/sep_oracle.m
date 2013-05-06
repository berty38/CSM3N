function [y, violation] = sep_oracle(w, kappa, labels, featureMap, S, scope)

n = length(labels);

ell = zeros(1,n);
ell(scope) = (2*(1 - labels(scope)) - 1)';

f = -w' * featureMap - ell;

if kappa > 0
    H = kappa * speye(n);
    [y, obj] = quadprog(H, f, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, S.x0, S.options);
else
    [y, obj] = linprog(f, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, S.x0, S.options);
end

violation = w' * featureMap * (y - labels) + sum(abs(y - labels)) + 0.5 * kappa * (labels'*labels - y' * y);

