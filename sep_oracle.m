function [y, violation] = sep_oracle(w, labels, featureMap, S, scope)

n = length(labels);

ell = zeros(1,n);
ell(scope) = (2*(1 - labels(scope)) - 1)';

f = -w' * featureMap - ell;

[y, obj] = linprog(f, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, S.x0);

violation = w' * featureMap * (y - labels) + sum(abs(y - labels));

