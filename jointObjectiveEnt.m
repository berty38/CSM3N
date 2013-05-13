function [f, g] = jointObjectiveEnt(x, F, labels, scope, S, C, z, F_labels)

% outputs the objective value and gradient of the joint learning objective
% using the dual of loss-augmented inference to make the objective a
% minimization
% scope is an index vector (or logical vector) indicating which entries of
% the marginal vector should be counted in the loss

if ~exist('F_labels', 'var')
    F_labels = F*labels;
end

[d,m] = size(F);

w = x(1:d);
kappa = max(0, x(d+1)); % don't let kappa be negative
lambda = x(d+2:end);

A = S.Aeq;
b = S.beq;

ell = zeros(size(labels));
ell(scope) = 1-labels(scope);
delta = sum(ell(scope));

y = exp((F'*w + ell + A'*lambda)/kappa - 1);

loss = kappa * sum(y) - b'*lambda + delta - w'*(F_labels);

f = 0.5*(w'*w)/sqrt(kappa + z) + C * loss;

if nargout == 2
    gradW = w/sqrt(kappa + z);
    gradW = gradW + C * ((F * y) - F_labels);
    gradKappa = - C*y(y>0)'*log(y(y>0)) - w'*w / (4*(kappa + z)^(3/2));
    gradLambda = C* (A * y - b);
    
    g = [gradW; gradKappa; gradLambda];
    
end