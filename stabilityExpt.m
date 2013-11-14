global mosek_path
global minFunc_path
mosek_path = '/Users/Ben/Code/mosek/6';
minFunc_path = '/Users/Ben/Code/MATLAB/';
initMosek;
initMinFunc;

clear;

% for testing
k = 2;
chainLength = 20;
numTest = 2;
totalRuns = 1;
% for experiment (comment out next few lines for testing)
k = 10;
chainLength = 200;
numTest = 20;
totalRuns = 20;

pObs = 0.2;
pSame = 0.9;

types = [1 2 3];

Cvec = 10.^linspace(-2,6,9);

maxSamp = 10;
nStabSamp = min(maxSamp, chainLength*(k-1));

scope = 1:chainLength*k;

total = length(types) * length(Cvec) * totalRuns;

totalTimer = tic;
count = 0;

%% Experiment
for run = 1:totalRuns
	
	%% generate chains
	
	[X,Y,A,pSame_tr(run)] = genMarkovChain(chainLength, k, pObs, pSame);
	
	for i = 1:numTest
		[Xte{i},Yte{i},Ate{i},pSame_te(run,i)] = genMarkovChain(chainLength, k, pObs, pSame);
	end
	
	
	%% generate overcomplete representation of ground truth
	
	labels = zeros(chainLength*k + nnz(A)*k^2, 1);
	for i = 1:chainLength
		labels(localIndex(i, Y(i), chainLength)) = 1;
	end
	
	[I,J] = find(A);
	for i = 1:nnz(A)
		labels(pairwiseIndex(i, Y(I(i)), Y(J(i)), chainLength, k)) = 1;
	end
	
	
	%% construct structural constraints
	
	[Aeq, beq, F_tr] = edge_marginals(X', A, k);
	
	S.Aeq = Aeq;
	S.beq = beq;
	S.A = [];
	S.b = [];
	S.lb = zeros(chainLength*k + nnz(A)*k^2, 1);
	S.ub = [];
	S.x0 = [];
	
	for i = 1:numTest
		[~, ~, F_te{i}] = edge_marginals(Xte{i}', Ate{i}, k);
	end
	
	
	%% compute true-weight predictions
	
% 	for i = 1:numTest
% 		w = handTune(pObs, pSame_te(i), k);
% 		y = crfInference(w, F_te{i}, chainLength*k, S);
% 		pred = predictMax(y(1:k*chainLength), chainLength, k);
% 		trueWeightErrorTe(i, run) = nnz(pred ~= Yte{i}) / chainLength;
% 	end
	
	%% compute base local error
	[Xpred, ~] = find(X');
	baseError(run) = nnz(Xpred~=Y) / length(Y);
	for i = 1:numTest
		[Xpred, ~] = find(Xte{i}');
		baseErrorTe(run, i) = nnz(Xpred~=Yte{i}) / chainLength;
	end
	
	
	%% Train/test
	for c=length(Cvec):-1:1
		for t=1:length(types)
			type = types(t);
			C = Cvec(c);
			
			% TRAINING
			if type == 1 % CRF
				w = learnCRF(F_tr, labels, chainLength*k, S, C);
				kappa = 1;
				y = crfInference(w, F_tr, chainLength*k, S);
			elseif type == 2 % M3N
				w = vanillaM3N(F_tr, labels, scope, S, C);
				kappa = 0;
				y = dualInference(w, F_tr, kappa, S);
			elseif type == 3 % CSM3N
				[w, kappa] = jointLearnEntLog(F_tr, labels, scope, S, C);
				y = dualInference(w, F_tr, kappa, S);
			end
			pred = predictMax(y(1:k*chainLength), chainLength, k);
			trainError(type, c, run) = nnz(pred ~= Y) / chainLength;
			
			% TESTING
			for i = 1:numTest
				if type == 1
					y = crfInference(w, F_te{i}, chainLength*k, S);
				else
					y = dualInference(w, F_te{i}, kappa, S);
				end
				pred = predictMax(y(1:k*chainLength), chainLength, k);
				testError(type, c, i, run) = nnz(pred ~= Yte{i}) / chainLength;
			end
			meanTestError(type, c, run) = mean(testError(type, c, :, run));
			varTestError(type, c, run) = var(testError(type, c, :, run));
			
			% STABILITY
			stab = zeros(numTest,2);
			for i = 1:numTest
				[stab(i,1),stab(i,2),stabSamp] = measureStabilityRand2(...
					w, X', k, F_te{i}, S, nStabSamp, type, kappa);
				savedStabSamp(c, type, run, i, 1:nStabSamp) = stabSamp;
			end
			maxStab = max(stab,[],1);
			savedStab(c, type, run, 1:2) = maxStab;
			
			% BOOKKEEPING
			savedW{c, type, run} = w;
			savedKappa(c, type, run) = kappa;
			
			count = count + 1;
			fprintf('Finished %d of %d, elapsed %f minutes, eta %f\n', ...
				count, total, toc(totalTimer)/60, ...
				(total - count)*(toc(totalTimer)/count)/60);
			
		end
	end
	
	save stabilityExptData.mat;
	
	%% Plot errors
	
	fig2 = figure(2);
	subplot(411);
	semilogx(Cvec, mean(baseError,2) * ones(size(Cvec)), '--ko');
	hold on;
	semilogx(Cvec, mean(trainError(types,:,:), 3), 'x-');
	hold off;
	title(sprintf('pSame=%.2f, pObs=%.2f', pSame, pObs));
	ylabel('Training error', 'FontSize', 14);
	xlabel('C', 'FontSize', 14);
	set(gca, 'FontSize', 14);
% 	legend('Local error', 'CRF', 'M3N');
	legend('Local error', 'CRF', 'M3N', 'CSM3N');

	subplot(412);
	semilogx(Cvec, mean(baseErrorTe(:)) * ones(size(Cvec)), '--ko');
	hold on;
	semilogx(Cvec, mean(meanTestError(types,:,:), 3), 'x-');
	hold off;
	%title(sprintf('pSameRange=[%.2f ... %.2f], pObs=%.2f', pSameMin, pSameMax, pObs));
	ylabel('Avg. testing error', 'FontSize', 14);
	xlabel('C', 'FontSize', 14);
	set(gca, 'FontSize', 14);
	
	subplot(413);
	semilogx(Cvec, mean(varTestError(types,:,:), 3), 'x-');
	xlabel('C', 'FontSize', 14);
	ylabel('Testing variance', 'FontSize', 14);
	set(gca, 'FontSize', 14);
	
	subplot(414);
	% this is probably the wrong way to compute generalization error...
	genError = meanTestError(types,:,:) - trainError(types,:,:);
	semilogx(Cvec, mean(genError,3), 'x-');
	title('Generalization error')
	xlabel('C', 'FontSize', 14);
	ylabel('Generalization', 'FontSize', 14);
	set(gca, 'FontSize', 14);

	%print(fig2, '-dpng', 'stabilityExpt.png');

	%% Plot weight norms
	
	figure(3)
	
	norms = zeros(length(Cvec),2);
	
	if ismember(1,types)
		subplot(411);
		for i = 1:length(Cvec)
			w = savedW{i, 1, run};
			norms(i,1) = norm(w(1:k^2));
			norms(i,2) = norm(w(k^2+1:end));
		end
		loglog(Cvec, norms.^2);
		xlabel('C');
		ylabel('||w||^2');
		title('Weight norms for CRF');
		legend('local', 'relational');
	end
	
	if ismember(2,types)
		subplot(412);
		for i = 1:length(Cvec)
			w = savedW{i, 2, run};
			norms(i,1) = norm(w(1:k^2));
			norms(i,2) = norm(w(k^2+1:end));
		end
		loglog(Cvec, norms.^2);
		xlabel('C');
		ylabel('||w||^2');
		title('Weight norms for M3N');
		legend('local', 'relational');
	end
	
	if ismember(3,types)
		subplot(413);
		for i = 1:length(Cvec)
			w = savedW{i, 3, run};
			norms(i,1) = norm(w(1:k^2));
			norms(i,2) = norm(w(k^2+1:end));
		end
		loglog(Cvec, norms.^2);
		xlabel('C');
		ylabel('||w||^2');
		title('Weight norms for CSM3N');
		legend('local', 'relational');

		subplot(414);
		loglog(Cvec, savedKappa(:,3,run))
		xlabel('C');
		ylabel('kappa');
		title('kappa for current run of CSM3N');
	end
	
	
	%% Plot stability
	figure(4)
	
	% get max stabilities
	stabs_mu = zeros(length(types),length(Cvec));
	stabs_y = zeros(length(types),length(Cvec));
	for i = 1:length(Cvec)
		stabs_mu(1,i) = max(savedStab(i, 1, :, 1));
		stabs_y(1,i) = max(savedStab(i, 1, :, 2));
	end
	for i = 1:length(Cvec)
		stabs_mu(2,i) = max(savedStab(i, 2, :, 1));
		stabs_y(2,i) = max(savedStab(i, 2, :, 2));
	end
	for i = 1:length(Cvec)
		stabs_mu(3,i) = max(savedStab(i, 3, :, 1));
		stabs_y(3,i) = max(savedStab(i, 3, :, 2));
	end
	
	% plot stability of marginals
	subplot(311);
	semilogx(Cvec, stabs_mu(types,:)/2, 'x-');
	xlabel('C', 'FontSize', 14);
	ylabel('1-norm / 2', 'FontSize', 14);
	title('Stability of marginals', 'FontSize', 14);
% 	legend('CRF', 'M3N');	
	legend('CRF', 'M3N', 'CSM3N');	

	% plot stability of decoding
	subplot(312);
	semilogx(Cvec, stabs_y(types,:), 'x-');
	xlabel('C', 'FontSize', 14);
	ylabel('Hamming norm', 'FontSize', 14);
	title('Stability of decoding', 'FontSize', 14);
	
	% plot marginal stability histogram
	% don't know how to do this yet...
	
% 	% scatter plot marginal stability
% 	stabs_scat = zeros(length(types),length(Cvec)*totalRuns);
% 	genErr_scat = zeros(length(types),length(Cvec)*totalRuns);
% % 	stabs_scat = zeros(length(types),length(Cvec));
% % 	genErr_scat = zeros(length(types),length(Cvec));
% 	for t=1:length(types)
% 		stabs_all = squeeze(savedStab(:, t, :, 1))/2;
% % 		stabs_all = squeeze(savedStab(:, t, run, 1))/2;
% 		stabs_scat(t,:) = stabs_all(:);
% 		genErr_all = squeeze(genError(t, :, :));
% % 		genErr_all = squeeze(genError(t, :, run));
% 		genErr_scat(t,:) = genErr_all(:);
% 	end
% % 	subplot(313)
% % 	plot(stabs_scat', genErr_scat', 'x');
% % 	xlabel('Marginal stability', 'FontSize', 14);
% % 	ylabel('Generalization error', 'FontSize', 14);
% % 	title('Stability vs. Generalization', 'FontSize', 14);
% 	subplot(413)
% 	plot(stabs_scat(1,:)', genErr_scat(1,:)', 'bo');
% 	xlabel('1-norm / 2', 'FontSize', 14);
% 	ylabel('gen error', 'FontSize', 14);
% 	title('Stability vs. Generalization CRF', 'FontSize', 14);
% 	subplot(414)
% 	plot(stabs_scat(2,:)', genErr_scat(2,:)', 'go');
% 	xlabel('1-norm / 2', 'FontSize', 14);
% 	ylabel('gen error', 'FontSize', 14);
% 	title('Stability vs. Generalization M3N', 'FontSize', 14);
	
		
end

