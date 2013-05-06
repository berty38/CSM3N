function [f, g, neg] = minmaxObjEnt(x, C, featureMap, labels, ell, z)

if nargin == 5
    z = 0;
end

d = size(featureMap, 1);

w = x(1:d);
kappa = x(d+1);
y = x(d+2:end);

kappa = 0;
% w = ones(size(w));
% y = 0.5*ones(size(y));

regularizer = (labels(labels>0)' * log(labels(labels>0)) - y(y>0)' * log(y(y>0)));

f = (w'*w) / (2 * sqrt(kappa + z)) + ...
    C * ((w'*featureMap) * (y - labels) + ...
    kappa * regularizer + ell' * y + nnz(labels));

gradW = w / sqrt(kappa + z) + C * featureMap * (y - labels);
gradKappa = C * regularizer - (w'*w) / (4 * (kappa + z)^1.5);
gradKappa = 0;
% gradW = 0*gradW;

if kappa > 0
    gradReg = kappa * C * (log(y) + 1);
else
    gradReg = zeros(size(y));
end
gradY = C * featureMap' * w - gradReg + C * ell;
% gradY = 0*gradY;

g = [gradW; gradKappa; gradY];

neg = [false(d+1,1); true(size(gradY))];
