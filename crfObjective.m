function [f,g] = crfObjective(x, F, S, C, singletons, Flabels, varargin)

[d,m] = size(F);

s = ones(m,1);
s(singletons+1:end) = 1;

w = x(1:d);
lambda = x(d+1:end);

y = exp(s .* (F'*w + S.Aeq'*lambda) - 1);

f = 0.5 * (w'*w) / C + sum(y) - w'*Flabels - S.beq'*lambda;

if nargout == 2
    gradW = w / C + F * (s .* y) - Flabels;
    gradLambda = S.Aeq * (s .* y) - S.beq;
    
    g = [gradW; gradLambda];
end
