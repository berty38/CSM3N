function [Aeq, beq, f_o, F] = triadMarginals(graph, obs)

% Generates the constraints associated with having
%   edges as local features and triads as relational features
% graph - (upper-triangular) adjacency matrix
% obs - observed edge values, in {-1,+1}
% Aeq, beq - pseudomarginal constraints
% f_o - features for observed variables
% F - feature map for unobserved variables

% Just in case, convert graph to upper-triangle
graph = triu(graph);
obs = trui(obs);

% Observed edges
[I,J,V] = find(obs);
obsEdges = [I J V];

% Unobserved edges
[I J] = find(graph-obs);
unoEdges = [I J];

% All edges
edges = [obsEdges(:,1:2) ; unoEdges];

% Triads
triads = findTriangles(graph);
obsTriads = findTriangle(obs);
unoTriads = setdiff(triads, obsTriads, 'rows');

% Dimensions.
n = size(graph,1);		% number nodes
n_e = size(edges,1);	% number edges
n_oe = size(obsEdges,1);
n_ue = size(unoEdges,1);
n_t = size(triads,1);	% number triads
n_ot = size(obsTriads,1);
n_ut = size(unoTriads,1);

% Create index of edges
edgeIdx = sparse(edges(:,1), edges(:,2), 1:n_e, n, n);


%% Observed features

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
	s = v1*4 + v2*2 + v1 + 1;
	f_tri_o(s) = f_tri_o(s) + 1;
end

% Al observed features
f_o = [f_loc_o ; f_tri_o];


%% Generate feature map
F_loc = sparse(eye(n_ue));
F_tri = zeros(2 * n_e);
for ut=1:n_ut
	i = unoTriads(ot,1);
	j = unoTriads(ot,2);
	k = unoTriads(ot,3);
	s = possibleTriadStates(obs(i,j), obs(i,k), obs(j,k));
	% TODO
end


%% Marginal constraints

AI = zeros(2*n_ue + 8*n_ut, 1);
AJ = zeros(2*n_ue + 8*n_ut, 1);
AV = zeros(2*n_ue + 8*n_ut, 1);
beq = zeros(n_ue + n_ut, 1);

nextA = 1;
nextb = 1;

% Local marginals must sum to 1
for ue=1:n_ue
	AI(nextA:nextA+1) = nextb;
	AJ(nextA:nextA+1) = nextA:nextA+1;
	AV(nextA:nextA+1) = 1;
	beq(nextb) = 1;
	nextA = nextA + 2;
	nextb = nextb + 1;
end

% Partially obs triad marginals must sum to 1
for ut=1:n_ut
	AI(nextA:nextA+7) = nextb;
	AJ(nextA:nextA+7) = triadIndex(ut,0,0,0,n_ue);
	AV(nextA:nextA+7) = 1;
	beq(nextb) = 1;
	nextA = nextA + 8;
	nextb = nextb + 1;
end

% Edge/triad agreement
for ut=1:n_ut
	% v1 = -1
	AI(nextA:nextA+3) = nextb;
	AJ(nextA:nextA+3) = triadIndex(ut,-1,0,0,n_ue);
	AV(nextA:nextA+3) = 1;
	AI(nextA+4) = nextb;
	AJ(nextA+4) = unoTriads(ut,1);
	AV(nextA+4) = -1;
	beq(nextb) = 0;
	nextA = nextA + 5;
	nextb = nextb + 1;
	% v1 = +1
	AI(nextA:nextA+3) = nextb;
	AJ(nextA:nextA+3) = triadIndex(ut,1,0,0,n_ue);
	AV(nextA:nextA+3) = 1;
	AI(nextA+4) = nextb;
	AJ(nextA+4) = unoTriads(ut,1);
	AV(nextA+4) = -1;
	beq(nextb) = 0;
	nextA = nextA + 5;
	nextb = nextb + 1;
	
end




