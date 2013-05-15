function [obsEdges, obsTriads, unoEdges, unoTriads] = setupTriads(graph, obs)

% Setup for edges as local features and triads as relational features.
% graph - (upper-triangular) adjacency matrix
% obs - observed edge values, in {-1,+1}
% obsEdges - List of observed edges in (i,j) pairs
% obsTriads - Sorted list of observed triads in (i,j,k) triples
% unoEdges - List of unobserved edges in (i,j) pairs
% unoTriads - Sorted list of unobserved triads in (i,j,k) triples

% Just in case, convert graph to upper-triangle
graph = triu(graph);
obs = triu(obs);

% Observed edges
[I,J,V] = find(obs);
obsEdges = [I J V];
% Unobserved edges
[I J] = find(graph-obs);
unoEdges = [I J];

% All triads
triads = findTriangles(graph);
% Observed Triads
obsTriads = findTriangles(obs);
% Partially-observed triads
if length(obsTriads) > 0
	unoTriads = setdiff(triads, obsTriads, 'rows');
else
	unoTriads = triads;
end


