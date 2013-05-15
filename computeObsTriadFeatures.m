function f_o = computeObsTriadFeatures(obsEdges, obsTriads)

% Generates the observed edge/triad features.
% obs - observed edge values, in {-1,+1}
% obsEdges - 
% Aeq, beq - pseudomarginal constraints
% f_o - features for observed variables
% F - feature map for unobserved variables

% Dimensions.
n_oe = size(obsEdges,1);
n_ot = size(obsTriads,1);


%% Precompute observed features

% Sum obs local features
f_loc_o = zeros(2,1);
for oe=1:n_oe
	v = obsEdges(oe,3);
	s = v/2 + 3/2;
	f_loc_o(s) = f_loc_o(s) + 1;
end

% Sum obs triad features
f_tri_o = zeros(8,1);
for ot=1:n_ot
	i = obsTriads(ot,1);
	j = obsTriads(ot,2);
	k = obsTriads(ot,3);
	v1 = (obs(i,j) + 1) / 2;
	v2 = (obs(i,k) + 1) / 2;
	v3 = obs(j,k);
	s = v1*4 + v2*2 + v3 + 1;
	f_tri_o(s) = f_tri_o(s) + 1;
end

% All observed features
f_o = [f_loc_o ; f_tri_o];

