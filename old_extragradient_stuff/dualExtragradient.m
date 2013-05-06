function [x, report] = dualExtragradient(fun, proj, x0, options)

% assumes fun returns [obj, gradient], where the gradient can be the
% adjusted saddle-point gradient of a convex-concave function

if ~exist('options', 'var')
    options = [];
end

if isfield(options, 'eta')
    eta = options.eta;
else
    eta = 0.2;
end

if isfield(options, 'tolerance')
    tolerance = options.tolerance;
else
    tolerance = 1e-6;
end

if isfield(options, 'maxiter')
    maxiter = options.maxiter;
else
    maxiter = 10000;
end

if isfield(options, 'callback')
    callback = options.callback;
else
    callback = [];
end

report = [];
converged = false;

obj = [];
iter = 1;
x = x0;

total = 0;

s = 0;

while ~converged && iter <= maxiter
    oldx = x;
    r = inf;
    
    total = total + x;
    
    % compute prediction
    xp = proj(x + eta * s);
    
    % compute correction step
    [~, gradc, neg] = fun(xp);
    gradc(neg) = -gradc(neg);
    x = proj(xp - eta * gradc);
    
    [~, grad, neg] = fun(xp);
    grad(neg) = -grad(neg);
    s = s - grad;
    
    change = norm(oldx - x);
    
    obj(iter) = fun(total / iter);
    
%     checkgrad(fun, x)
    
    % run callback
    if ~isempty(callback)
        report = callback(total/iter, obj, report);
    end
    
    if change < tolerance
        converged = true;
    end
    
    iter = iter + 1;
end

x = total / (iter - 1);