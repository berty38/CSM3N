function [Aeq, beq, F] = triadMarginals(obs, edges, triads, edgeIdx, triadIdx)

% Generates the marginal constraints and feature map for
%   edges as local features and triads as relational features.
% obs - observed edge values, in {-1,+1}
% edges - list of edges (i,j)
% triads = list of triads (i,j,k)
% Aeq, beq - pseudomarginal constraints
% F - feature map for unobserved variables

% Dimensions.
n_e = size(edges,1);
n_t = size(triads,1);
n_y = triadIdx(end);
n_y_t = triadIdx(end) - triadIdx(1);


%% Generate feature map

FI = [];
FJ = [];
FV = [];
nextF = 1;

% Local feature map
FI(1:2*n_e) = repmat([1;2], n_e, 1);
FJ(1:2*n_e) = 1:2*n_e;
FV(1:2*n_e) = 1;
nextF = nextF + 2*n_e;

% Triad feature map
for t=1:n_t
	i = triads(t,1);
	j = triads(t,2);
	k = triads(t,3);
	states = triadStates(obs(i,j), obs(i,k), obs(j,k));
	FI(nextF:nextF+length(states)-1) = 2 + states;
	FJ(nextF:nextF+length(states)-1) = triadIdx(t,1):triadIdx(t,2);
	FV(nextF:nextF+length(states)-1) = 1;
	nextF = nextF + length(states);
end

% Create F
F = sparse(FI, FJ, FV, 10, n_y);


%% Marginal constraints

% For efficiency, we'll preallocate the constraint arrays using a ballpark
% size.
n_beq = n_e + n_t + n_y_t;
n_Aeq = n_y + 4*n_y_t;
AI = zeros(n_Aeq,1);
AJ = zeros(n_Aeq,1);
AV = zeros(n_Aeq,1);
beq = zeros(n_beq,1);
nextA = 1;
nextb = 1;

% Local marginals must sum to 1
for e=1:n_e
	AI(nextA:nextA+1) = nextb;
	AJ(nextA:nextA+1) = nextA:nextA+1;
	AV(nextA:nextA+1) = 1;
	beq(nextb) = 1;
	nextA = nextA + 2;
	nextb = nextb + 1;
end

% Triad marginals must sum to 1
for t=1:n_t
	idx = triadIdx(t,1):triadIdx(t,2);
	AI(nextA:nextA+length(idx)-1) = nextb;
	AJ(nextA:nextA+length(idx)-1) = idx;
	AV(nextA:nextA+length(idx)-1) = 1;
	nextA = nextA + length(idx);
	beq(nextb) = 1;
	nextb = nextb + 1;
end

% Edge/triad agreement
for t=1:n_t
	i = triads(t,1);
	j = triads(t,2);
	k = triads(t,3);
	idx = triadIdx(t,1):triadIdx(t,2);
	allStates = triadStates(obs(i,j), obs(i,k), obs(j,k));

	% v1
	if obs(i,j) == 0
		% v1 = -1
		[~,states] = ismember(triadStates(-1, obs(i,k), obs(j,k)), allStates);
		AI(nextA:nextA+length(states)-1) = nextb;
		AJ(nextA:nextA+length(states)-1) = idx(states);
		AV(nextA:nextA+length(states)-1) = 1;
		AI(nextA+length(states)) = nextb;
		AJ(nextA+length(states)) = 2*edgeIdx(i,j)-1;
		AV(nextA+length(states)) = -1;
		beq(nextb) = 0;
		nextA = nextA + length(states) + 1;
		nextb = nextb + 1;
		% v1 = +1
		[~,states] = ismember(triadStates(1, obs(i,k), obs(j,k)), allStates);
		AI(nextA:nextA+length(states)-1) = nextb;
		AJ(nextA:nextA+length(states)-1) = idx(states);
		AV(nextA:nextA+length(states)-1) = 1;
		AI(nextA+length(states)) = nextb;
		AJ(nextA+length(states)) = 2*edgeIdx(i,j);
		AV(nextA+length(states)) = -1;
		beq(nextb) = 0;
		nextA = nextA + length(states) + 1;
		nextb = nextb + 1;
	end
	
	% v2
	if obs(i,k) == 0
		% v2 = -1
		[~,states] = ismember(triadStates(obs(i,j), -1, obs(j,k)), allStates);
		AI(nextA:nextA+length(states)-1) = nextb;
		AJ(nextA:nextA+length(states)-1) = idx(states);
		AV(nextA:nextA+length(states)-1) = 1;
		AI(nextA+length(states)) = nextb;
		AJ(nextA+length(states)) = 2*edgeIdx(i,k)-1;
		AV(nextA+length(states)) = -1;
		beq(nextb) = 0;
		nextA = nextA + length(states) + 1;
		nextb = nextb + 1;
		% v2 = +1
		[~,states] = ismember(triadStates(obs(i,j), 1, obs(j,k)), allStates);
		AI(nextA:nextA+length(states)-1) = nextb;
		AJ(nextA:nextA+length(states)-1) = idx(states);
		AV(nextA:nextA+length(states)-1) = 1;
		AI(nextA+length(states)) = nextb;
		AJ(nextA+length(states)) = 2*edgeIdx(i,k);
		AV(nextA+length(states)) = -1;
		beq(nextb) = 0;
		nextA = nextA + length(states) + 1;
		nextb = nextb + 1;
	end
	
	% v3
	if obs(j,k) == 0
		% v3 = -1
		[~,states] = ismember(triadStates(obs(i,j), obs(i,k), -1), allStates);
		AI(nextA:nextA+length(states)-1) = nextb;
		AJ(nextA:nextA+length(states)-1) = idx(states);
		AV(nextA:nextA+length(states)-1) = 1;
		AI(nextA+length(states)) = nextb;
		AJ(nextA+length(states)) = 2*edgeIdx(j,k)-1;
		AV(nextA+length(states)) = -1;
		beq(nextb) = 0;
		nextA = nextA + length(states) + 1;
		nextb = nextb + 1;
		% v3 = +1
		[~,states] = ismember(triadStates(obs(i,j), obs(i,k), 1), allStates);
		AI(nextA:nextA+length(states)-1) = nextb;
		AJ(nextA:nextA+length(states)-1) = idx(states);
		AV(nextA:nextA+length(states)-1) = 1;
		AI(nextA+length(states)) = nextb;
		AJ(nextA+length(states)) = 2*edgeIdx(j,k);
		AV(nextA+length(states)) = -1;
		beq(nextb) = 0;
		nextA = nextA + length(states) + 1;
		nextb = nextb + 1;
	end
end

% Create Aeq, beq
beq = beq(1:nextb-1);
Aeq = sparse(AI(1:nextA-1), AJ(1:nextA-1), AV(1:nextA-1), length(beq), n_y);

