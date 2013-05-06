function [f, g] = jointObjectiveEnt(x, F, labels, scope, S, C, z)

% outputs the objective value and gradient of the joint learning objective
% using the dual of loss-augmented inference to make the objective a
% minimization
% scope is an index vector (or logical vector) indicating which entries of
% the marginal vector should be counted in the loss

[d,m] = size(F);

w = x(1:d);
kappa = x(d+1);
lambda = x(d+2:end);

A = S.Aeq;
b = S.beq;

ell = zeros(size(labels));
ell(scope) = 1-labels(scope);
delta = sum(ell(scope));

y = exp((F'*w + ell + A'*lambda)/kappa - 1);

labelEnt = labels(labels>0)'*log(labels(labels>0));
loss = kappa * sum(y) - b'*lambda + delta - w'*F*labels + kappa * labelEnt;

f = 0.5*(w'*w)/sqrt(kappa + z) + C * loss;

gradW = w/sqrt(kappa + z) + C * F * (y - labels);
gradKappa = C*labelEnt - C*y(y>0)'*log(y(y>0)) - w'*w / (4*(kappa + z)^(3/2));
gradLambda = C* (A * y - b);

g = [gradW; gradKappa; gradLambda];

