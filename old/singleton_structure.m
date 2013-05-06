function [Aeq, beq, F] = singleton_structure(X, graph, k)

% generates the constraints associated with having edge marginals over all
% pairs of k labels
% X binary local features of nodes (each column is a node)
% graph - adjacency matrix of network
% k - number of categories
% Aeq, beq - pseudomarginal constraints
% F feature map

[d,n] = size(X);


beq = zeros(n + nnz(graph) + 2 * nnz(graph) * k, 1);
next = 1;

assert(all(size(graph) == [n n]));

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
localF = sparse(LI, LJ, LV, d * k, n * k);


AI = [];
AJ = [];
AV = [];

% set up local marginal constraints
for i = 1:n
    inds = localIndex(i, 1:k, n);
    AI(end+1:end+k) = next;
    AJ(end+1:end+k) = inds;
    AV(end+1:end+k) = 1;
    %   Aeq(next, inds) = 1;
    beq(next) = 1;
    
    next = next + 1;
end


% set up pairwise features

PI = zeros(2*k^2,1);
PJ = zeros(2*k^2,1);
PV = zeros(2*k^2,1);

for i = 1:k
    for j = 1:k
        
    end
end
