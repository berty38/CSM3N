function [f, g] = jointObjectiveEntLog(x, F, labels, scope, S, C, F_labels, varargin)

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
logkappa = x(d+1);
lambda = x(d+2:end);

A = S.Aeq;
b = S.beq;

ell = zeros(size(labels));
ell(scope) = 1-2*labels(scope);
% delta = sum(labels(scope));

FtwlAl = (F'*w + ell + A'*lambda);

logy = FtwlAl/exp(logkappa) - 1;
sumy = exp(logkappa + logy);
y = exp(logy);

loss = sum(sumy) - w'*F_labels - b'*lambda;

f = 0.5*(w'*w) / (C * exp(2*logkappa)) + loss;
% f = 0.5*(w'*w) / C + 1 / (2*C*exp(2*logkappa)) + loss;

if nargout == 2
    gradW = w / (C*exp(2*logkappa)) + (F * y) - F_labels;
%     gradW = w / C + (F * y) - F_labels;
    gradLambda = A * y - b;
    gradKappa =  - w'*w * exp(-2*logkappa) / C +  y'*(exp(logkappa) - FtwlAl);
%     gradKappa =  - exp(-2*logkappa) / C +  y'*(exp(logkappa) - FtwlAl);
    
    g = [gradW; gradKappa; gradLambda];
end
