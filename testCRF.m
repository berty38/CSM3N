initMinFunc
clear;

m = 10;
n = 10;
d = 100;

F = randn(d,m);

labels = zeros(m,1);
labels(5) = 1;

S.Aeq = ones(n,m);
S.beq = ones(n,1);

singletons = 3;

x = zeros(d+n,1);

C = .1;

func = @(x) crfObjective(x, F, S, C, singletons, F*labels);

derivativeCheck(func, x, 1, 0);

[w, x0] = learnCRF(F, labels, singletons, S, C);

derivativeCheck(func, x0, 1, 0);

y = crfInference(w, F, singletons, S, C);

