%% Setup
clear
global mosek_path
global minFunc_path
mosek_path = '/Users/Ben/Code/mosek/6';
minFunc_path = '/Users/Ben/Code/MATLAB/';
initMosek()
initMinFunc()

%% Data

% create ground truth
% n = 100;
% p = 0.2;
% gt = sign(sprandn(n, n, p));
loadEpinions();

% create graph rep
graph = abs(gt);

%% Experiments

n_fold = 1;

for fold=1:n_fold
	
	% Snowball sample train/test
	[g_tr, g_te] = snowballSample(graph);
	gt_tr = gt .* g_tr;
	gt_te = gt .* g_te;

	%% Setup train
	
	% Unobserve some edges values
	obs = gt_tr;
	idx = find(obs);
	idx = randsample(idx, floor(0.5*length(idx)));
	obs(idx) = 0;

	% Setup experiment
	[obsEdges, obsTriads, unoEdges, unoTriads] = setupTriads(g_tr, obs);
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
		if gt_tr(i,j) < 0
			y(2*ue-1) = 1;
		else
			y(2*ue) = 1;
		end
	end
	for ut=1:n_t
		i = unoTriads(ut,1);
		j = unoTriads(ut,2);
		k = unoTriads(ut,3);
		idx = triadIndex(unoTriadIdx,unoTriads,obs,ut,gt_tr(i,j),gt_tr(i,k),gt_tr(j,k));
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
	S_tr.A = [];
	S_tr.b = [];
	S_tr.Aeq = Aeq;
	S_tr.beq = beq;
	S_tr.lb = zeros(n_y,1);
	S_tr.ub = [];
	S_tr.x0 = [];
	
% 	w = vanillaM3N(F, y, scope, S, C);
	[w, kappa] = jointLearnEnt(F, y, scope, S_tr, C);

	%% Evaluation
	yhat = dualInference(w, F, kappa, S_te);

end



