function [X,Y,A] = generateMarkovChain(n, k, pSame, pObs)

I = 1:n-1;
J = 2:n;
A = sparse(I,J,ones(n-1,1), n, n);

X = zeros(n, k);

Y = zeros(n,1);

Y(1) = randi(k);
X(1, Y(1)) = 1;

for i = 2:n
    
    same = rand < pSame;
    
    if same
        Y(i) = Y(i-1);
    else
        Y(i) = randi(k);
    end
    
    obs = rand < pObs;
    
    if obs
        X(i,Y(i)) = 1;
    else
        X(i, randi(k)) = 1;
    end
end
