function y = inference(w, kappa, featureMap, S) 

n = size(featureMap, 2);

f = -w' * featureMap;


if kappa > 0
    H = kappa * speye(n);
    y = quadprog(H, f, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, S.x0, S.options);
else
    y = linprog(f, S.A, S.b, S.Aeq, S.beq, S.lb, S.ub, S.x0, S.options);
end
