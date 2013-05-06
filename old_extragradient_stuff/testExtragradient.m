clear;
initMosek;

n = 200;
d = 6000;
k = 4;

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
%%

C = 1;

nu = 1;
scope = 1:n*k;

[w, kappa, y] = extragradientLearn(featureMap, labels, C, S, scope, nu);

pred = inference(w, featureMap, kappa, S);

fprintf('primal training error %f\n', norm(pred - labels));


% [w2, kappa2, y2] = dualExtragradientLearn(featureMap, labels, C, S, scope, nu);
% 
% pred = inference(w2, kappa2, featureMap, S);
% 
% fprintf('dual training error %f\n', norm(pred - labels));


%%
% 
% ell = zeros(size(labels));
% ell(scope) = 1 - labels(scope);
% 
% fun = @(x) minmaxObjEnt(x, C, featureMap, labels, ell)
% 
% 
% x = rand(k * d + k^2 + length(labels) + 1, 1);
% % x = [w; kappa; y];
% 
% checkgrad(fun, x)
% 
% [f,g] = fun(x);
% 
% 
