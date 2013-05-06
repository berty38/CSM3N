function violation = checkMarginal(y, graph, k)

% for debugging, checks that a vector is in the pseudomarginal polytope

n = size(graph, 1);

violation = 0;

local = 0;
pairwise = 0;
marginal = 0;

tol = 1e-5;

for i = 1:n
    inds = localIndex(i, 1:k, n);
    violation = violation + abs(sum(y(inds)) - 1);
    local = local + (abs(sum(y(inds)) - 1) > tol);
end

[I,J] = find(graph);

for i = 1:nnz(graph)
    
    inds = pairwiseIndex(i, 1:k, 1:k, n, k);
    
    violation = violation + abs(sum(y(inds)) - 1);
    pairwise = pairwise + (abs(sum(y(inds)) - 1) > tol);
    
    
    
    for c = 1:k
        
        inds = pairwiseIndex(i, 1:k, c, n, k);
        
        violation = violation + abs(sum(y(inds)) - y(localIndex(J(i), c, n)));
        marginal = marginal + (abs(sum(y(inds)) - y(localIndex(J(i), c, n))) > tol);
        
        inds = pairwiseIndex(i, c, 1:k, n, k);
        
        violation = violation + abs(sum(y(inds)) - y(localIndex(I(i), c, n)));
        marginal = marginal + (abs(sum(y(inds)) - y(localIndex(I(i), c, n))) > tol);

    end
end

fprintf('%d local marginals are not on the simplex\n', local);
fprintf('%d pairwise marginals are not on the simplex\n', pairwise);
fprintf('%d marginals are not consist\n', marginal);
