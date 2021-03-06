function idx = triadIndex(triadIdx, triads, obs, t, v1, v2, v3)

% Returns the indices of the triad terms.
% triadIdx - Tx2 table of triad indices (computed by getTriadIndex)
% traids - Tx3 table of (i,j,k) triads
% obs - observations
% t  - index of the t'th triad
% v1 - state of first var
% v2 - state of second var
% v3 - state of third var
%   NOTE: Assumes states in {-1,0,+1}, where 0 means any state.
% idx - list of indices to variables

i = triads(t,1);
j = triads(t,2);
k = triads(t,3);
o1 = obs(i,j);
o2 = obs(i,k);
o3 = obs(j,k);
allIdx = triadIdx(t,1):triadIdx(t,2);
allStates = triadStates(o1, o2, o3);
state = triadStates(v1, v2, v3);
[check,stateIdx] = ismember(state, allStates);
if nnz(check) ~= length(state)
	fprintf('Error: triad: %d, obs: (%d,%d,%d), state: (%d,%d,%d)\n',...
		t, full(o1),full(o2),full(o3), full(v1),full(v2),full(v3));
	idx = -1;
else
	idx = allIdx(stateIdx);
end

