clear;
initMosek;

n = 10;
d = 20;
k = 3;

trueW = randn(k * d + k^2,1);

graph = rand(n,n) < 2/n;

X = rand(d,n) < .2;
%%
fprintf('Setting up QP relaxation\n');
[Aeq, beq, featureMap] = edge_marginals(X, graph, k);
fprintf('Set up QP relaxation\n');


S.A = [];
S.b = [];
S.Aeq = Aeq;
S.beq = beq;
S.lb = zeros(n * k + nnz(graph) * k^2,1);
S.ub = [];
S.x0 = [];
S.options = optimset('algorithm', 'interior-point-convex', 'display', 'off');

labels = inference(trueW, 0, featureMap, S);

checkMarginal(labels, graph, k)

%%

C = 1;

kappa = 1.00;
[w, kappa, violation, xi] = stableSVMStruct(featureMap, labels, S, C, kappa, true);
