function inds = triadIndex(i, c1, c2, c3, n, k)

% Returns the indices of the triad terms.
% i  - index of the ith triad
% c1 - state of first var
% c2 - state of second var
% c3 - state of third var
% n  - number of local variables
% k  - number of states per variable

state = bsxfun(@plus, (c1-1).*k^2, bsxfun(@plus, (c2-1).*k, c3'));
inds = n*k + (i-1)*k^3 + state;
inds = inds(:);

