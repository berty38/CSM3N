%% workspace to help me derive the dual to loss-augmented inference


%% set up test problem
clear;
initMosek;

n = 5;
d = 3;
k = 3;

violation = inf;

while violation > 1e-6
    
    trueW = randn(k * d + k^2,1);
    %
    % graph = rand(n,n) < 2/n;
    graph = diag(ones(n-1,1),1);
    graph = graph + graph';
    
    X = rand(d,n) < .2;
    
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
    
    labels = euclideanProject(S, rand(n * k + nnz(graph) * k^2, 1));
    
    violation = checkMarginal(labels, graph, k)
end

scope = false(size(labels));
scope(1:k*d) = true;

%% check primal

w = randn(k*d + k^2, 1);
kappa = rand;

ell = zeros(size(labels));
ell(scope) = 1 - labels(scope);

[yp, objp] = inference(w, featureMap, kappa, S, [], ell);
[yd, objd] = dualInference(w, featureMap, kappa, S, ell);
yp-yd

[objp objd]

%% 

x = randn(k*d+k^2+1+size(S.Aeq,1), 1);
x(k*d+k^2+1) = rand;
%%
C = 2.4;

func = @(y) jointObjectiveEnt(y, featureMap, labels, scope, S, C, 0.1);

checkgrad(func, x)

%% 
clear options;
options.Display = 'iter';
options.Algorithm = 'interior-point';
options.GradObj = 'on';
options.TolFun = 1e-16;
options.TolCon = 1e-16;
options.TolX = 1e-16;
options.MaxFunEvals = inf;
options.MaxIter = inf;
% fminconOptions.Diagnostics = 'on';
% fminconOptions.DerivativeCheck = 'on';

lb = -inf(k*d+k^2+1+size(S.Aeq,1), 1);
lb(k*d+k^2+1) = 0;

x = fmincon(func, x, [], [], [], [], lb, [], [], options);



%%

[w, kappa, y] = jointLearnEnt(featureMap, labels, scope, S, 1);

