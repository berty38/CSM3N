initMinFunc
clear;

m = 10;
n = 10;
d = 10;

F = randn(d,m);

labels = zeros(m,1);
labels(5) = 1;

scope = true(m,1);
S.Aeq = ones(n,m);
S.beq = ones(n,1);

x = zeros(d+n+1,1);
x(d+1) = 1;

C = 100;

func = @(y) jointObjectiveEnt(y, F, labels, scope, S, C, F*labels);

derivativeCheck(func, x, 1, 0);

[w, kappa, y, x0] = jointLearnEnt(F, labels, scope, S, C, x);

derivativeCheck(func, x0, 1, 0);

y2 = dualInference(w, F, kappa, S);

