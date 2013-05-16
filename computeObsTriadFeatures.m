function f_o = computeObsTriadFeatures(obs, edges, triads)

% Generates the observed edge/triad features.
% obs - observed edge values, in {-1,+1}
% edges - list of edges (i,j)
% triads = list of triads (i,j,k)
% f_o - features for observed variables

% Dimensions.
n_e = size(edges,1);
n_t = size(triads,1);


%% Precompute observed features

f_o = zeros(10,1);

% Sum obs local features
for e=1:n_e
	v = edges(e,3);
	s = v/2 + 3/2;
	f_o(s) = f_o(s) + 1;
end

% Sum obs triad features
for t=1:n_t
	i = triads(t,1);
	j = triads(t,2);
	k = triads(t,3);
	s = triadStates(obs(i,j), obs(i,k), obs(j,k));
	f_o(2+s) = f_o(2+s) + 1;
end


