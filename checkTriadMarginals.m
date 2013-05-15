clear

% create ground truth
gt = sparse(8,8);
gt(1,2)=-1;
gt(1,3)=-1;
gt(1,5)=1;
gt(2,5)=1;
gt(4,6)=-1;
gt(6,7)=1;
gt(2,8)=-1;
gt(5,8)=1;

% unobserve some edges values
obs = gt;
obs(2,5)=0;
obs(2,8)=0;

% create graph rep
graph = abs(gt);

% setup
[obsEdges, obsTriads, unoEdges, unoTriads] = setupTriads(graph, obs);
[unoEdgeIdx, unoTriadIdx] = genTriadIndex(obs, unoEdges, unoTriads);
f_o = computeObsTriadFeatures(obsEdges, obsTriads);
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

% check marginal constraints
fprintf('Distance to constraint satisfaction: %f\n', sum(Aeq*y-beq));


