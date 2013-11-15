function [f, g] = vctsmObj(x, F, labels, scope, S, C, F_labels, varargin)

% Outputs the objective value and gradient of the VCTSM learning objective
% using the dual of loss-augmented inference to make the objective a
% minimization.
% Scope is an index vector (or logical vector) indicating which entries of
% the marginal vector should be counted in the loss.

if ~exist('F_labels', 'var')
    F_labels = F*labels;
end

[d,m] = size(F);

w = x(1:d);
logkappa = x(d+1);
lambda = x(d+2:end);

A = S.Aeq;
b = S.beq;

delta = sum(labels(scope));
ell = zeros(size(labels));
ell(scope) = 1-2*labels(scope);
wtw = w' * w;
z = (F'*w + ell + m*A'*lambda);
y = exp( exp(-logkappa)*z - 1 );

loss = (exp(logkappa)*sum(y) - w'*F_labels) / m - lambda'*b;

f = 0.5 * exp(-2*logkappa) * C * wtw + loss;

if nargout == 2
    gradW = exp(-2*logkappa) * C * w + F * y - F_labels;
    gradKappa = -exp(-2*logkappa) * C * wtw +  y' * (exp(logkappa) - z) / m;
    gradLambda = A * y - b;
    g = [gradW; gradKappa; gradLambda];
end
