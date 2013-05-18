function [y, obj] = inference(w, featureMap, kappa, S, y0, ell)

% run inference. If kappa is nonzero, use minFunc to solve dual. If kappa
% is zero, solve as primal linear program

[~,m] = size(featureMap);

if kappa > 0
    if exist('y0', 'var') && ~isempty(y0)
        y = y0;
    else
        y = euclideanProject(S, 0.5 * ones(m,1));
    end
    
    if exist('ell', 'var')
        fun = @(y) inferenceObjEnt(y, w, featureMap, kappa, ell);
    else
        fun = @(y) inferenceObjEnt(y, w, featureMap, kappa);
    end
    
    clear fminconOptions;
    fminconOptions.Display = 'notify';
    fminconOptions.Algorithm = 'interior-point';
    fminconOptions.GradObj = 'on';
    fminconOptions.Hessian = 'lbfgs';
    fminconOptions.TolFun = 1e-9;
    fminconOptions.TolCon = 1e-9;
    [y, obj] = fmincon(fun, y, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, [], fminconOptions);
    obj = -obj; % since we are supposed to be maximizing
else
    options.Display = 'notify';
    options.MSK_IPAR_INTPNT_NUM_THREADS = 4;
    [y, obj] = linprog(-w'*featureMap, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, S.x0, options);
end

