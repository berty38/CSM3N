function [x, report] = extragradient(fun, proj, x0, options)

% assumes fun returns [obj, gradient], where the gradient can be the
% adjusted saddle-point gradient of a convex-concave function

if ~exist('options', 'var')
    options = [];
end

if isfield(options, 'nu')
    nu = options.nu;
else
    nu = 1;
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
beta = 1;

obj = [];
iter = 1;
x = x0;

while ~converged && iter <= maxiter
    oldx = x;
    r = inf;
    
    while r >= nu
        % compute prediction
        [obj(iter), grad, neg] = fun(x);
        grad(neg) = -grad(neg);
        xp = proj(x - beta * grad);
        
        % compute correction step
        [~, gradc, neg] = fun(xp);
        gradc(neg) = -gradc(neg);
        
        % check step size
        gradDiff = grad - gradc;
        diff = x - xp;
        
        r = beta * (gradDiff'*gradDiff) / sqrt( diff'*diff);
        
        if r >= nu
            beta = (2/3) * beta * min(1, 1/r);
        end
    end
    
    % update variables
    x = proj(x - beta * gradc);
    change = norm(oldx - x);
   
    % run callback
    if ~isempty(callback)
        report = callback(x, obj, report);
    end
    
    if change < tolerance
        converged = true;
    end
    
    iter = iter + 1;
end

