function [f, g] = minmaxObj(x, C, featureMap, labels, ell)

d = size(featureMap, 1);

w = x(1:d);
kappa = x(d+1);
y = x(d+2:end);

gradW = w / (C * (kappa + 1)) + featureMap * (y - labels);
gradKappa = (labels'*labels - y'*y) / 2 - w'*w / (2 * C * (kappa + 1)^2);
gradY = featureMap' * w - kappa * y + ell;
% gradY = 0 * gradY;
%gradW = 0 * gradW;
% gradKappa = 0 * gradKappa;

g = [gradW; gradKappa; gradY];

f = (w' * w) / (2 * C * (kappa + 1)) + w'* featureMap * (y - labels) + ...
    (kappa / 2) * (labels' * labels - y' * y) + ell' * y + nnz(labels);
