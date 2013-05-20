function [S, F, y, scope] = setupEpinions(graph, gt, pctObs)

% Unobserve some edges values
obs = gt;
idx = find(obs);
idx = randsample(idx, floor(pctObs*length(idx)));
obs(idx) = 0;

% Setup graph stuff and marginal constraints
[edges, triads] = setupTriads(graph, obs);
[edgeIdx, triadIdx] = genTriadIndex(obs, edges, triads);
[Aeq, beq, F] = triadMarginals(obs, edges, triads, edgeIdx, triadIdx);

% Create ground truth marginal vector
n_e = size(edges,1);
n_t = size(triads,1);
n_y = triadIdx(end);
y = zeros(n_y,1);
for ue=1:n_e
	i = edges(ue,1);
	j = edges(ue,2);
	if gt(i,j) < 0
		y(2*ue-1) = 1;
	else
		y(2*ue) = 1;
	end
end
for ut=1:n_t
	i = triads(ut,1);
	j = triads(ut,2);
	k = triads(ut,3);
	idx = triadIndex(triadIdx,triads,obs,ut,gt(i,j),gt(i,k),gt(j,k));
	y(idx) = 1;
end

% Sanity checks
fprintf('Size of Aeq: %d x %d\n', size(Aeq,1), size(Aeq,2));
fprintf('Size of beq: %d x %d\n', size(beq,1), size(beq,2));
fprintf('Size of F:   %d x %d\n', size(F,1), size(F,2));
fprintf('Size of y:   %d x %d\n', size(y,1), size(y,2));
fprintf('Distance to constraint satisfaction: %f\n', sum(Aeq*y-beq));

scope = 1:2*n_e;
S.A = [];
S.b = [];
S.Aeq = Aeq;
S.beq = beq;
S.lb = zeros(n_y,1);
S.ub = [];
S.x0 = [];

