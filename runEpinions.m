%% Setup
clear
% global mosek_path
% global minFunc_path
% mosek_path = '/Users/blondon/Code/mosek/6';
% minFunc_path = '/Users/blondon/Code/MATLAB/';
% initMosek()
% initMinFunc()

%% Data

% create ground truth
% n = 500;
% p = 0.2;
% gt = triu(sign(sprandn(n, n, p)));
loadEpinions();

% create graph rep
graph = abs(gt);

%% Experiments

n_fold = 1;

for fold=1:n_fold
	
	%% Setup
	
	% Snowball sample train/test
	[g_tr, g_te] = snowballSample(graph);
	gt_tr = gt .* g_tr;
	gt_te = gt .* g_te;

	% Setup train
	
	[S_tr, F_tr, y_tr, scope_tr] = setupEpinions(g_tr, gt_tr, 0.5);
	
	% Setup test
	
	[S_te, F_te, y_te, scope_te] = setupEpinions(g_te, gt_te, 0.5);
		
	%% Experiment
	
% 	Cvals = 10.^linspace(0,5,6);
% 	methods = [1 2];
% 	for c=1:length(Cvals)
% 		for m=1:length(methods)
% 		
% 			% Training
% 			if methods(m) == 1 % M3N
%                 w = vanillaM3N(F_tr, y_tr, scope_tr, S_tr, Cvals(c));
%                 kappa = 0;
%                 y = dualInference(w, F_tr, kappa, S_tr);
% 			else if methods(m) == 2 % CSM3N
%                 [w, kappa] = jointLearnEntLog(F_tr, y_tr, scope_tr, S_tr, 10*Cvals(c));
%                 y = dualInference(w, F_tr, kappa, S_tr);
% 			else
% 				fprintf('Unsupported method: %d\n', methods(m));
% 				continue
% 			end
% 			pred = predictMax(y(1:k*chainLength), chainLength, k);
% 			trainError(type, cIndex, run) = nnz(pred ~= Y) / chainLength;
% 						
% 			% Testing
% 		end
% 	end
% 
end



