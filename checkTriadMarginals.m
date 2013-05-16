clear

% create ground truth
n = 100;
p = 0.2;
gt = sign(sprandn(n, n, p));

% unobserve some edges values
obs = gt;
idx = find(obs);
idx = randsample(idx, floor(0.5*length(idx)));
obs(idx) = 0;

% create graph rep
graph = abs(gt);

% setup
[obsEdges, obsTriads, unoEdges, unoTriads] = setupTriads(graph, obs);
[unoEdgeIdx, unoTriadIdx] = genTriadIndex(obs, unoEdges, unoTriads);
f_o = computeObsTriadFeatures(obs, obsEdges, obsTriads);
[Aeq, beq, F] = triadMarginals(obs, unoEdges, unoTriads, unoEdgeIdx, unoTriadIdx);

% create marginal vector using ground truth
n_y = unoTriadIdx(end);
y = zeros(n_y,1);
for ue=1:size(unoEdges,1)
	i = unoEdges(ue,1);
	j = unoEdges(ue,2);
	if gt(i,j) < 0
		y(2*ue-1) = 1;
	else
		y(2*ue) = 1;
	end
end
for ut=1:size(unoTriads,1)
	i = unoTriads(ut,1);
	j = unoTriads(ut,2);
	k = unoTriads(ut,3);
	idx = triadIndex(unoTriadIdx,unoTriads,obs,ut,gt(i,j),gt(i,k),gt(j,k));
	y(idx) = 1;
end

% check dimensions
fprintf('Size of Aeq: %d x %d\n', size(Aeq,1), size(Aeq,2));
fprintf('Size of beq: %d x %d\n', size(beq,1), size(beq,2));
fprintf('Size of F:   %d x %d\n', size(F,1), size(F,2));
fprintf('Size of y:   %d x %d\n', size(y,1), size(y,2));

% check feature map
f_o
Fy = F * y

% check marginal constraints
fprintf('Distance to constraint satisfaction: %f\n', sum(Aeq*y-beq));


