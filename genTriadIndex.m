function [edgeIdx, triadIdx] = genTriadIndex(obs, edges, triads)

% Generates indices used for un-/partially-observed edges/triads

% Dimensions
n = size(obs, 1);
n_e = size(edges, 1);
n_t = size(triads, 1);

% Generate index of edge predictions
edgeIdx = sparse(edges(:,1), edges(:,2), 1:n_e, n, n);

% Generate linear index of triads
triadIdx = zeros(n_t,2); % each line is [BEGINNING, END] of indices
prevEnd = 2*n_e;
for t=1:n_t
	i = triads(t,1);
	j = triads(t,2);
	k = triads(t,3);
	states = triadStates(obs(i,j), obs(i,k), obs(j,k));
	triadIdx(t,:) = [prevEnd+1, prevEnd+length(states)];
	prevEnd = prevEnd + length(states);
end

