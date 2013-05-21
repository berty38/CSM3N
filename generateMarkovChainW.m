function [X,Y,A] = generateMarkovChainW(n, k, w)

I = 1:n-1;
J = 2:n;
A = sparse(I,J,ones(n-1,1), n, n);

pObs = reshape(w(1:k^2), [k k]);
pTrans = reshape(w(k^2+1:end), [k k]);

% X = zeros(n, k+1);
X = zeros(n, k);

Y = zeros(n,1);

i = 1;
% generate first node
Y(i) = randi(k);    
X(i, sampleFromTable(pObs(Y(i),:))) = 1;


for i = 2:n
    Y(i) = sampleFromTable(pTrans(Y(i-1),:));
    
    X(i, sampleFromTable(pObs(Y(i),:))) = 1;
end


function i = sampleFromTable(w)

p = exp(w);
p = p / sum(p);
cdf = cumsum(p);
i = find(rand < cdf, 1, 'first');
