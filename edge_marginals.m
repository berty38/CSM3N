function [Aeq, beq, F] = edge_marginals(X, graph, k)

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


edgeF = zeros(k^2, nnz(graph) * k^2);

% set up pseudomarginal constraints and overcomplete features
[I,J] = find(graph);
for i = 1:nnz(graph)
    inds = pairwiseIndex(i, 1:k, 1:k, n, k);
    
    edgeF(:,inds - n * k) = eye(k^2);
    
    % make marginal sum to 1
    
    AI(end+1:end+k^2) = next;
    AJ(end+1:end+k^2) = inds;
    AV(end+1:end+k^2) = 1;
    beq(next) = 1;
    next = next + 1;
    
    for val = 1:k
        % marginalize over a
        inds = pairwiseIndex(i, 1:k, val, n, k);        
        AI(end+1:end+k) = next;
        AJ(end+1:end+k) = inds;
        AV(end+1:end+k) = 1;

        AI(end+1) = next;
        AJ(end+1) = localIndex(J(i), val, n);
        AV(end+1) = -1;
        beq(next) = 0;
        
        next = next + 1;
        
        % marginalize over b
        inds = pairwiseIndex(i, val, 1:k, n, k);
        AI(end+1:end+k) = next;
        AJ(end+1:end+k) = inds;
        AV(end+1:end+k) = 1;

        AI(end+1) = next;
        AJ(end+1) = localIndex(I(i), val, n);
        AV(end+1) = -1;
        beq(next) = 0;
        
        next = next + 1;
    end
    
end

Aeq = sparse(AI, AJ, AV, n + nnz(graph) + 2 * nnz(graph) * k, n * k + nnz(graph) * k^2);

F = [localF sparse(k * d, nnz(graph) * k^2); sparse(k^2, n*k) edgeF];

