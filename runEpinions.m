%% Setup
clear
global mosek_path
global minFunc_path
mosek_path = '/Users/blondon/Code/mosek/7';
minFunc_path = '/Users/blondon/Code/MATLAB/';
initMosek()
initMinFunc()

%% Data

% create ground truth
% n = 100;
% p = 0.2;
% gt = sign(sprandn(n, n, p));
loadEpinions()
n = size(gt,1);

% create graph rep
graph = abs(gt);

%% Experiments

n_fold = 1;

for fold=1:n_fold
	%% Seup
	
	% Unobserve some edges values
	obs = gt;
	idx = find(obs);
	idx = randsample(idx, floor(0.5*length(idx)));
	obs(idx) = 0;

	% Setup experiment
	[obsEdges, obsTriads, unoEdges, unoTriads] = setupTriads(graph, obs);
	[unoEdgeIdx, unoTriadIdx] = genTriadIndex(obs, unoEdges, unoTriads);
	f_o = computeObsTriadFeatures(obs, obsEdges, obsTriads);
	[Aeq, beq, F] = triadMarginals(obs, unoEdges, unoTriads, unoEdgeIdx, unoTriadIdx);

	% Create ground truth marginal vector
	n_e = size(unoEdges,1);
	n_t = size(unoTriads,1);
	n_y = unoTriadIdx(end);
	y = zeros(n_y,1);
	for ue=1:n_e
		i = unoEdges(ue,1);
		j = unoEdges(ue,2);
		if gt(i,j) < 0
			y(2*ue-1) = 1;
		else
			y(2*ue) = 1;
		end
	end
	for ut=1:n_t
		i = unoTriads(ut,1);
		j = unoTriads(ut,2);
		k = unoTriads(ut,3);
		idx = triadIndex(unoTriadIdx,unoTriads,obs,ut,gt(i,j),gt(i,k),gt(j,k));
		y(idx) = 1;
	end

	% Sanity checks
	fprintf('Size of Aeq: %d x %d\n', size(Aeq,1), size(Aeq,2));
	fprintf('Size of beq: %d x %d\n', size(beq,1), size(beq,2));
	fprintf('Size of F:   %d x %d\n', size(F,1), size(F,2));
	fprintf('Size of y:   %d x %d\n', size(y,1), size(y,2));
	fprintf('Distance to constraint satisfaction: %f\n', sum(Aeq*y-beq));
	
	%% Let's do some learnin'!
	C = 1;
	scope = 1:2*n_e;
	S.A = [];
	S.b = [];
	S.Aeq = Aeq;
	S.beq = beq;
	S.lb = zeros(n_y,1);
	S.ub = [];
	S.x0 = [];
	
% 	w = vanillaM3N(F, y, scope, S, C);
	w = jointLearnEnt(F, y, scope, S, C);

end


