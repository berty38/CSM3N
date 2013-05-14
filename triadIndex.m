function idx = triadIndex(triad, c1, c2, c3, n_single)

% Returns the indices of the triad terms.
% i  - index of the ith triad
% c1 - state of first var
% c2 - state of second var
% c3 - state of third var
% n  - number of local variables
% NOTE: Assumes states in {-1,0,+1}, where 0 means any state

% state = bsxfun(@plus, (c1-1).*k^2, bsxfun(@plus, (c2-1).*k, c3'));
state = possibleTriadStates(c1, c2, c3);
idx = 2*n_single + 8*(triad-1) + state;

