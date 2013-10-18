function F = localFeatures(X,k)

% X is a d x n matrix, representing n nodes taking d values.
% k is the number of labels in the hidden states.

[d,n] = size(X);

[I,J,V] = find(X);

LI = zeros(nnz(X) * k, 1);
LJ = zeros(nnz(X) * k, 1);
LV = zeros(nnz(X) * k, 1);

for val = 1:k
    startJ = localIndex(1, val, n) - 1;
    startI = (val - 1) * d ;
  
    LI((val-1)*nnz(X) + 1:val*nnz(X)) = I + startI;
    LJ((val-1)*nnz(X) + 1:val*nnz(X)) = J + startJ;
    LV((val-1)*nnz(X) + 1:val*nnz(X)) = V;
end
F = sparse(LI, LJ, LV, d * k, n * k);

