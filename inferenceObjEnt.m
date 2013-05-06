function [f,g] = inferenceObjEnt(y, w, featureMap, kappa, ell)

% inference or loss-augmented inference objective and gradient

regularizer = y(y>0)' * log(y(y>0));

f = -w' * featureMap * y + kappa * regularizer;

g = -featureMap' * w + kappa * (log(y) + 1);

if exist('ell', 'var')
    f = f - ell' * y;
    g = g - ell;
end



