function [X,Y,A] = generateMarkovChain(n, k, pSame, pObs)

I = 1:n-1;
J = 2:n;
A = sparse(I,J,ones(n-1,1), n, n);

%X = zeros(n, k + 1);
X = zeros(n,k);

Y = zeros(n,1);

i = 1;
% generate first node
Y(i) = randi(k);
obs = rand < pObs;
if obs
    X(i,Y(i)) = 1;
else
    %X(i, randi(k+1)) = 1;
    X(i,randi(k)) = 1;
end

% classProbs = 1:k;
% classProbs = classProbs / sum(classProbs);
classProbs = ones(k,1)/k; % uniform
classCDF = cumsum(classProbs);

for i = 2:n    
    same = rand < pSame;
    
    if same
        Y(i) = Y(i-1);
    else
        Y(i) = find(rand < classCDF, 1, 'first');
    end
    
    obs = rand < pObs;
    
    if obs
        X(i,Y(i)) = 1;
    else
        %X(i, randi(k+1)) = 1;
        X(i,randi(k)) = 1;
    end
end
