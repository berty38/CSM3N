function [y, obj, status] = euclideanProject(S, yc)

% finds the closest point in constraints defined in S

% set infinite values in yc to some large number
% yc(yc > 1e16) = 1e16;

m = length(yc);

H = speye(m);

f = -yc;

options = S.options;

% options.maxIter = 5;

[y, obj, status] = quadprog(H, f, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, S.x0, options);

