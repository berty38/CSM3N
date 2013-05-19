initMinFunc
initMosek
clear;

m = 1; % length of y vector
n = 1; % number of constraints
d = 1; % number of potentials

F = randn(d,m);

labels = zeros(m,1);
labels(randi(m)) = 1;

scope = true(m,1);
S.Aeq = ones(n,m);
S.beq = ones(n,1);

x = zeros(d+n+1,1);
x(d+1) = 0;

C = 5;
kappa = 1;

func = @(y) jointObjectiveEntLog(y, F, labels, scope, S, C, F*labels, kappa);

[f, g] = jointObjectiveEntLog(x, F, labels, scope, S, C, F*labels, kappa)

derivativeCheck(func, x, 1, 0);

[w, kappa, y, x0] = jointLearnEntLog(F, labels, scope, S, C, x, kappa);

derivativeCheck(func, x0, 1, 0);

y2 = dualInference(w, F, kappa, S);

