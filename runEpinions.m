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

n_runs = 1;
Cvals = 10.^linspace(0,5,6);
methods = [1 2];
trainStats = zeros(length(methods),length(Cvals),2,n_runs);
testStats = zeros(length(methods),length(Cvals),2,n_runs);

for run=1:n_runs
	
	fprintf('Starting run %d\n', run);
	
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
	
	for c=1:length(Cvals)
		for m=1:length(methods)
			fprintf('Running method %d with C=%.1f ... ', methods(m), Cvals(c));
			
			% Training
			if methods(m) == 1 % M3N
                w = vanillaM3N(F_tr, y_tr, scope_tr, S_tr, Cvals(c));
                kappa = 0;
                y = dualInference(w, F_tr, kappa, S_tr);
			elseif methods(m) == 2 % CSM3N
                [w, kappa] = jointLearnEntLog(F_tr, y_tr, scope_tr, S_tr, 10*Cvals(c));
                y = dualInference(w, F_tr, kappa, S_tr);
			else
				fprintf('Unsupported method: %d\n', methods(m));
				continue
			end
			[~,acc,f1,f1class] = singletonStats(y_tr(scope_tr), y(scope_tr), 2);
			trainStats(methods(m),Cvals(c),:,run) = [acc f1class(2)];
						
			% Testing
			y = dualInference(w, F_te, kappa, S_te);
			[~,acc,f1,f1class] = singletonStats(y_te(scope_te), y(scope_te), 2);
			testStats(methods(m),Cvals(c),:,run) = [acc f1class(2)];
			
			fprintf('done.\n', methods(m), Cvals(c));
		end
		
		% Log results for C
		fprintf('Results for run %d with C=%.1f:\n', run, Cvals(c));
		for m=1:length(methods)
			fprintf('  Method %d (train): acc: %f, F1: %f\n',...
				methods(m),...
				trainStats(methods(m),Cvals(c),1,run),...
				trainStats(methods(m),Cvals(c),2,run));
			fprintf('  Method %d (test): acc: %f, F1: %f\n',...
				methods(m),...
				testStats(methods(m),Cvals(c),1,run),...
				testStats(methods(m),Cvals(c),2,run));
		end
	end

end



