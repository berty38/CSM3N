function inds = localIndex(i, c, n)

% returns the index (or indices) of variable i being assigned to c, where
% there are k states per local variable, and n variables total

inds = (c - 1) .* n + i;