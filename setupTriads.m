function [edges, triads] = setupTriads(graph, obs)

% Setup for edges as local features and triads as relational features.
%   Ignores fully-observed edge/triad potentials.
% graph - (upper-triangular) adjacency matrix
% obs - observed edge values, in {-1,+1}
% edges - List of unobserved edges in (i,j) pairs
% triads - Sorted list of partially-observed triads in (i,j,k) triples

% Just in case, convert graph to upper-triangle
graph = triu(graph);
obs = triu(abs(obs));

% Unobserved edges
[I J] = find(graph-obs);
edges = [I J];

% All triads
triads = findTriangles(graph);
% Observed Triads
obsTriads = findTriangles(obs);
% Partially-observed triads
if size(obsTriads,1) > 0
	triads = setdiff(triads, obsTriads, 'rows');
end


