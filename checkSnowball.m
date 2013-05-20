
% Generate a graph
n = 1000;
p = 0.1;
graph = sparse(triu(rand(n) < p));

% Snowball sample
[train, test] = snowballSample_bak(graph);

% Check that all edges collected
edges = [train ; test];
graph_ = sparse(edges(:,1), edges(:,2), ones(size(edges,1),1), n, n);
diff = abs(graph - graph_);
fprintf('Number different edges: %d\n', nnz(diff));
