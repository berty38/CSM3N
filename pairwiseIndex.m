function inds = pairwiseIndex(i, c1, c2, n, k)

% returns the indices of the pairwise terms. the i'th edge in the graph,
% where [I,J] = find(graph), 
% c1 - the state of variable I(i)
% c2 - the state of variable J(i)
% n - number of local variables
% k - number of states per variable

inds = n * k + (i - 1) * k^2 + bsxfun(@plus, (c1-1).* k, c2');
inds = inds(:);

