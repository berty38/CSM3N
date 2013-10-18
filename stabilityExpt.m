global mosek_path
global minFunc_path
mosek_path = '/Users/blondon/Code/mosek/6';
minFunc_path = '/Users/blondon/Code/MATLAB/';
initMosek;
initMinFunc;

clear;

k = 3;
chainLength = 200;

pObs = 0.2;
pSameMin = 0.2;
pSameMax = 0.9;

numTest = 10;
totalRuns = 10;

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
	
	[X,Y,A,pSame_tr(run)] = genMarkovChain(chainLength, k, pObs, pSameMin, pSameMax);
	
	for i = 1:numTest
		[Xte{i},Yte{i},Ate{i},pSame_te(run,i)] = genMarkovChain(chainLength, k, pObs, pSameMin, pSameMax);
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
		figure(1);
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
			stab = zeros(numTest,1);
			for i = 1:numTest
% 				stab(i) = measureStability(w, X', k, F_te{i}, S, type, kappa);
% 				stab(i) = measureStabilityRand1(w, X', k, F_te{i}, S, type, kappa)
				stab(i) = measureStabilityRand2(w, X', k, F_te{i}, S, nStabSamp, type, kappa);
			end
			maxStab = max(stab);
			savedStab(c, type, run) = maxStab;
			
			% BOOKKEEPING
			savedW{c, type, run} = w;
			savedKappa(c, type, run) = kappa;
			
			count = count + 1;
			fprintf('Finished %d of %d, elapsed %f minutes, eta %f\n', ...
				count, total, toc(totalTimer)/60, ...
				(total - count)*(toc(totalTimer)/count)/60);
			
		end
	end
	
	%save markovSynthResultsShort;
	
	%% Plot errors
	
	fig2 = figure(2);
	subplot(411);
	semilogx(Cvec, mean(baseError,2) * ones(size(Cvec)), '--ko');
	hold on;
	semilogx(Cvec, mean(trainError(types,:,:), 3), 'x-');
	hold off;
	title(sprintf('pSameRange=[%.2f ... %.2f], pObs=%.2f', pSameMin, pSameMax, pObs));
	ylabel('Training error', 'FontSize', 14);
	xlabel('C', 'FontSize', 14);
	set(gca, 'FontSize', 14);
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
	
	norms = zeros(length(Cvec),1);
	
	if ismember(1,types)
		subplot(411);
		for i = 1:length(Cvec)
			norms(i) = norm(savedW{i, 1, run});
		end
		loglog(Cvec, norms.^2);
		ylabel('||w||^2');
		xlabel('C');
		title('norm for CRF');
	end
	
	if ismember(2,types)
		subplot(412);
		for i = 1:length(Cvec)
			norms(i) = norm(savedW{i, 2, run});
		end
		loglog(Cvec, norms.^2);
		ylabel('||w||^2');
		xlabel('C');
		title('norm for M3N');
	end
	
	if ismember(3,types)
		subplot(413);
		for i = 1:length(Cvec)
			norms(i) = norm(savedW{i, 3, run});
		end
		loglog(Cvec, norms.^2);
		ylabel('||w||^2');
		xlabel('C');
		title('norm for CSM3N');

		subplot(414);
		loglog(Cvec, savedKappa(:,3,run))
		ylabel('kappa');
		xlabel('C');
		title('kappa for current run of CSM3N');
	end
	
	
	%% Plot stability
	figure(4)
	
	stabs = zeros(length(Cvec),1);
	
	if ismember(1,types)
		subplot(311);
		for i = 1:length(Cvec)
			stabs(i) = max(savedStab(i, 1, :));
		end
		semilogx(Cvec, stabs, 'x-');
		ylabel('stability');
		xlabel('C');
		title('stability of CRF');
	end
	
	if ismember(2,types)
		subplot(312);
		for i = 1:length(Cvec)
			stabs(i) = max(savedStab(i, 2, :));
		end
		semilogx(Cvec, stabs, 'x-');
		ylabel('stability');
		xlabel('C');
		title('stability of M3N');
	end
	
	if ismember(3,types)
		subplot(313);
		for i = 1:length(Cvec)
			stabs(i) = max(savedStab(i, 3, :));
		end
		semilogx(Cvec, stabs, 'x-');
		ylabel('stability');
		xlabel('C');
		title('stability of CSM3N');
	end
	
		
end

