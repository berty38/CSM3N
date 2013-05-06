function [w, kappa, y, report] = extragradientLearn(featureMap, labels, C, S, scope, nu)

[d, m] = size(featureMap);

ell = zeros(m,1);
ell(scope) = 1 - labels(scope);

initY = euclideanProject(S, rand(m,1));%labels + ones(m,1));
initKappa = 0;

z = 1;

fun = @(x) minmaxObjEnt(x,C,featureMap,labels,ell,z);
proj = @(x) project(x, S, d);

x0 = [zeros(d, 1); initKappa; initY];

options = [];
options.nu = nu;
[U,S,V] = svds(featureMap);
normF = max(diag(S));
options.eta = 1/(C*normF);
options.callback = @(x, obj, storage) learningCallback(x, obj, storage, featureMap, labels);
options.maxiter = 600;

% [x2, report2] = extragradient(fun, proj, x0, options);
[x, report] = dualExtragradient(fun, proj, x0, options);

w = x(1:d);
kappa = x(d+1);
y = x(d+2:end);



function x = project(x, S, d)
 
decay = 0.5;
persistent minKappa;
if isempty(minKappa)
    minKappa = 0;
end

% project kappa
if x(d+1) < minKappa
    x(d+1) = minKappa;
    minKappa = minKappa * decay;
%     fprintf('Shrinking min kappa to %d\n', minKappa);
end
    

x(d+2:end) = euclideanProject(S, x(d+2:end));
% x(d+2:end) = max(0, min(x(d+2:end), 1));

