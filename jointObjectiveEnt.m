function [f, g] = jointObjectiveEnt(x, F, labels, scope, S, C, F_labels, varargin)

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

%isolate w
% kappa = 1;

A = S.Aeq;
b = S.beq;

ell = zeros(size(labels));
ell(scope) = 1-2*labels(scope);
% delta = sum(labels(scope));

logy = (F'*w + ell + A'*lambda)/kappa - 1;
y = exp(logy);

loss = C * (kappa * sum(y) - w'*F_labels - b'*lambda);

f = 0.5*(w'*w) / (kappa^2) + loss;

maxG = 1e16;

if nargout == 2
    gradW = w / (kappa^2) + C*(F * y) - C*F_labels; 
    gradKappa =  - w'*w / (kappa^3) -  C*y(y>0)'*logy(y>0);
    gradLambda = C * (A * y - b);

    g = [gradW; gradKappa; gradLambda];
    
%    g(g>maxG) = maxG;
%    g(g<-maxG) = -maxG;    
end
