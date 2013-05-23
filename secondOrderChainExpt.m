global mosek_path
global minFunc_path
mosek_path = '/Users/blondon/Code/mosek/6';
minFunc_path = '/Users/blondon/Code/MATLAB/';
initMosek;
initMinFunc;

clear;

k = 3;

chainLength = 300;
chainLengthTe = 300;
p2Hop = 0.5;
pSame = [0.7 0.9];
pObs = 0.2;

numTest = 10;
totalRuns = 5;

types = [1 2 3];

Cvec = 10.^linspace(-2,6,9);

scope = 1:chainLength*k;

total = length(types) * length(Cvec) * totalRuns;

totalTimer = tic;
count = 0;

%% Experiment
for run = 1:totalRuns
	
	%% generate chains
	
	[X,Y,A] = genSecondOrderChain(chainLength, k, p2Hop, pSame, pObs);
	
	for i = 1:numTest
		[Xte{i},Yte{i},Ate{i}] = genSecondOrderChain(chainLength, k, p2Hop, pSame, pObs);
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
		[AeqTe, beqTe, F_te{i}] = edge_marginals(Xte{i}', Ate{i}, k);
		Ste{i}.Aeq = Aeq;
		Ste{i}.beq = beq;
		Ste{i}.A = [];
		Ste{i}.b = [];
		Ste{i}.lb = zeros(chainLengthTe*k + nnz(Ate{i})*k^2, 1);
		Ste{i}.ub = [];
		Ste{i}.x0 = [];
	end
	
	
	%% compute true-weight predictions
	
% 	w = handTune(pObs, pSame_te, k);
% 	for i = 1:numTest
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
			
			if type == 1 % CRF
				w = learnCRF(F_tr, labels, chainLength*k, S, C);
				kappa = 1;
				y = crfInference(w, F_tr, chainLength*k, S);
			elseif type == 2 % M3N
				w = vanillaM3N(F_tr, labels, scope, S, C);
				kappa = 0;
				y = dualInference(w, F_tr, kappa, S);
			elseif type == 3 % CSM3N k=1
				[w, kappa] = jointLearnEntLog(F_tr, labels, scope, S, C, [], 1);
				y = dualInference(w, F_tr, kappa, S);
			else % CSM3N
                [w, kappa] = jointLearnEntLog(F_tr, labels, scope, S, C);
                y = dualInference(w, F_tr, kappa, S);
			end
			
			pred = predictMax(y(1:k*chainLength), chainLength, k);
			
			trainError(type, c, run) = nnz(pred ~= Y) / chainLength;
			
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
			
			savedW{c, type, run} = w;
			savedKappa(c, type, run) = kappa;
			
			count = count + 1;
			fprintf('Finished %d of %d, elapsed %f minutes, eta %f\n', count,...
				total, toc(totalTimer)/60, (total - count)*(toc(totalTimer)/count)/60);
			
		end
	end
	
	%save markovSynthResultsShort;
	
	%%
	figure(2);
	subplot(311);
	semilogx(Cvec, mean(baseError,2) * ones(size(Cvec)), '--ko');
	hold on;
	semilogx(Cvec, mean(trainError, 3), 'x-');
	hold off;
	% axis([min(Cvec), max(Cvec), 0, 1])
	title(sprintf('p2Hop=%.2f, pSame=[%.2f,%.2f], pObs=%.2f', p2Hop, pSame(1), pSame(2), pObs));
	ylabel('Training error', 'FontSize', 14);
	xlabel('C', 'FontSize', 14);
	set(gca, 'FontSize', 14);
	legend('Local error', 'CRF', 'M3N', 'CSM3N k=1');%, 'CSM3N');
	
	subplot(312);
	semilogx(Cvec, mean(baseErrorTe(:)) * ones(size(Cvec)), '--ko');
	hold on;
	semilogx(Cvec, mean(meanTestError, 3), 'x-');
	hold off;
	% axis([min(Cvec), max(Cvec), 0, 1])
	title(sprintf('p2Hop=%.2f, pSame=[%.2f,%.2f], pObs=%.2f', p2Hop, pSame(1), pSame(2), pObs));
	ylabel('Avg. testing error', 'FontSize', 14);
	xlabel('C', 'FontSize', 14);
	set(gca, 'FontSize', 14);
	
	subplot(313);
	semilogx(Cvec, mean(varTestError, 3), 'x-');
	xlabel('C', 'FontSize', 14);
	ylabel('Testing variance', 'FontSize', 14);
	set(gca, 'FontSize', 14);
	
	%%
	figure(3)
	
% 	subplot(411);
% 	
% 	loglog(Cvec, savedKappa(:,end,run))
% 	ylabel('kappa');
% 	xlabel('C');
% 	title('kappa for current run of CSM3N');
	
	norms = zeros(length(Cvec),1);
% 	subplot(412);
	subplot(311);
	for i = 1:length(Cvec)
		norms(i) = norm(savedW{i, 1, run});
	end
	loglog(Cvec, norms.^2);
	ylabel('||w||^2');
	xlabel('C');
	title('norm for CRF');
	
% 	subplot(413);
	subplot(312);
	for i = 1:length(Cvec)
		norms(i) = norm(savedW{i, 2, run});
	end
	loglog(Cvec, norms.^2);
	ylabel('||w||^2');
	xlabel('C');
	title('norm for M3N');
	
% 	subplot(414);
	subplot(313);
	for i = 1:length(Cvec)
		norms(i) = norm(savedW{i, 3, run});
	end
	loglog(Cvec, norms.^2);
	ylabel('||w||^2');
	xlabel('C');
	title('norm for CSM3N k=1');
	
end

