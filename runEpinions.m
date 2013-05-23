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
Cvals = 10;%10.^linspace(0,5,6);
methods = [1 2];
trainStats = zeros(length(methods),length(Cvals),5,n_runs);
testStats = zeros(length(methods),length(Cvals),5,n_runs);

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
                [w, kappa] = jointLearnEntLog(F_tr, y_tr, scope_tr, S_tr, Cvals(c));
                y = dualInference(w, F_tr, kappa, S_tr);
			else
				fprintf('Unsupported method: %d\n', methods(m));
				continue
			end
			[~,acc,f1,f1class] = singletonStats(y_tr(scope_tr), y(scope_tr), 2);
			labels = overcomplete2label(y_tr(scope_tr),[0;1]);
			preds = y(2:2:length(scope_tr));
			[~,~,~,auc] = perfcurve(labels, preds, 1);
			trainStats(methods(m),Cvals(c),:,run) = [acc f1 f1class' auc];
						
			% Testing
			y = dualInference(w, F_te, kappa, S_te);
			[~,acc,f1,f1class] = singletonStats(y_te(scope_te), y(scope_te), 2);
			labels = overcomplete2label(y_te(scope_te),[0;1]);
			preds = y(2:2:length(scope_te));
			[~,~,~,auc] = perfcurve(labels, preds, 1);
			testStats(methods(m),Cvals(c),:,run) = [acc f1 f1class' auc];
			
			fprintf('done.\n', methods(m), Cvals(c));
		end
		
		% Log results for C
		fprintf('Results for run %d with C=%.1f:\n', run, Cvals(c));
		for m=1:length(methods)
			fprintf('  Method %d (train): acc: %f, F1: %f F1-: %f, F1+: %f, AUC: %f\n',...
				methods(m),...
				trainStats(methods(m),Cvals(c),1,run),...
				trainStats(methods(m),Cvals(c),2,run),...
				trainStats(methods(m),Cvals(c),3,run),...
				trainStats(methods(m),Cvals(c),4,run),...
				trainStats(methods(m),Cvals(c),5,run));
			fprintf('  Method %d (test): acc: %f, F1: %f F1-: %f, F1+: %f, AUC: %f\n',...
				methods(m),...
				testStats(methods(m),Cvals(c),1,run),...
				testStats(methods(m),Cvals(c),2,run),...
				testStats(methods(m),Cvals(c),3,run),...
				testStats(methods(m),Cvals(c),4,run),...
				testStats(methods(m),Cvals(c),5,run));
		end
	end

end



