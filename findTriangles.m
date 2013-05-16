function triads = findTriangles(graph)

% Finds all (undirected) triangles in a graph.
% graph - (upper-triangular) adjacency matrix
% triads - T x 3 matrix, where T is the number of triads

% convert graph to upper-triangular (if not already so)
graph = triu(graph);
[I,J] = find(graph);
IJ = sortrows([I J]);
T = size(IJ,1);
triads = [];

for ij=1:(T-2)
	i = IJ(ij,1);
	j = IJ(ij,2);
	ik = ij + 1;
	while ik <= size(IJ,1) && IJ(ik,1) == i
		k = IJ(ik,2);
		if graph(j,k)
			triads = [triads ; i j k];
		end
		ik = ik + 1;
	end
end



