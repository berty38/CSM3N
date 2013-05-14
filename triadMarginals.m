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
% All edges (ordered by observed, unobserved)
edges = [obsEdges(:,1:2) ; unoEdges];

% All triads
triads = findTriangles(graph);
% Observed Triads
obsTriads = findTriangle(obs);
% Partially-observed triads
unoTriads = setdiff(triads, obsTriads, 'rows');

% Dimensions.
n = size(graph,1);		% number nodes
n_e = size(edges,1);	% number edges
n_oe = size(obsEdges,1);
n_ue = size(unoEdges,1);
n_t = size(triads,1);	% number triads
n_ot = size(obsTriads,1);
n_ut = size(unoTriads,1);


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
	s = v1*4 + v2*2 + v1 + 1;
	f_tri_o(s) = f_tri_o(s) + 1;
end

% All observed features
f_o = [f_loc_o ; f_tri_o];


%% Generate indices used for un-/partially-observed edges/triads

% Generate index of edge predictions
edgeIdx = sparse(unoEdges(:,1), unoEdges(:,2), 1:n_ue, n, n);

% Generate linear index of triad predictions
triadIdx = zeros(n_ut,2); % each line is [BEGINNING, END] of indices
prevEnd = 2*n_ue;
for ut=1:n_ut
	i = unoTriads(ut,1);
	j = unoTriads(ut,2);
	k = unoTriads(ut,3);
	states = triadStates(obs(i,j), obs(i,k), obs(j,k));
	triadIdx(ut,:) = [prevEnd+1, prevEnd+length(states)];
	prevEnd = prevEnd + length(states);
end


%% Generate feature map

FI = [];
FJ = [];
FV = [];
nextF = 1;

% Local feature map is just identity.
FI(1:2*n_ue) = 1:2*n_ue;
FJ(1:2*n_ue) = 1:2*n_ue;
FV(1:2*n_ue) = 1;
nextF = nextF + 2*n_ue;

% Triad feature map
for ut=1:n_ut
	i = unoTriads(ut,1);
	j = unoTriads(ut,2);
	k = unoTriads(ut,3);
	states = triadStates(obs(i,j), obs(i,k), obs(j,k));
	sIdx = triadIdx(ut,1);
	nextState = 0;
	for s=1:8
		if ismember(s, states)
			FI(nextF) = 2*n_ue + 8*(ut-1) + s;
			FJ(nextF) = sIdx + nextState;
			FV(nextF) = 1;
			nextF = nextF + 1;
			nextState = nextState + 1;
		end
	end
end

% Create F
F = sparse(FI, FJ, FV, 2*n_ue + 8*n_ut, triadIdx(end));


%% Marginal constraints

AI = [];
AJ = [];
AV = [];
beq = [];
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

% Triad marginals must sum to 1
for ut=1:n_ut
	i = unoTriads(ut,1);
	j = unoTriads(ut,2);
	k = unoTriads(ut,3);
	states = triadStates(obs(i,j), obs(i,k), obs(j,k));
	sIdx = triadIdx(ut,1);
	eIdx = triadIdx(ut,2);
	AI(nextA:nextA+length(states)-1) = nextb;
	AJ(nextA:nextA+length(states)-1) = sIdx:eIdx;
	AV(nextA:nextA+length(states)-1) = 1;
	nextA = nextA + length(states);
	beq(nextb) = 1;
	nextb = nextb + 1;
end

% Edge/triad agreement
for ut=1:n_ut
	i = unoTriads(ut,1);
	j = unoTriads(ut,2);
	k = unoTriads(ut,3);

	% v1
	if obs(i,j) == 0
		% v1 = -1
		states = triadStates(-1, obs(i,k), obs(j,k));
		AI(nextA:nextA+length(states)-1) = nextb;
		AJ(nextA:nextA+length(states)-1) = states;
		AV(nextA:nextA+length(states)-1) = 1;
		AI(nextA+length(states)) = nextb;
		AJ(nextA+length(states)) = 2*edgeIdx(i,j)-1;
		AV(nextA+length(states)) = -1;
		beq(nextb) = 0;
		nextA = nextA + length(states) + 1;
		nextb = nextb + 1;
		% v1 = +1

	end
	
	% v2
	if obs(i,k) == 0
		% v2 = -1
		
		% v2 = +1
		
	end
	
	% v3
	if obs(j,k) == 0
		% v3 = -1
		
		% v3 = +1
		
	end
end




