initMosek;
initMinFunc;

clear;

chainLength = 50;
pObs = .2;
pSame = .8;

numTest = 10;
totalRuns = 20;

k = 3;

scope = 1:chainLength*k;

Cvec = 10.^linspace(-3,5,15);
% Cvec = [1e-2 1e-1];


for run = 1:totalRuns
    
    %% generate chains
    
    [X,Y,A] = generateMarkovChain(chainLength, k, pSame, pObs);
    
    for i = 1:numTest
        [Xte{i},Yte{i},Ate{i}] = generateMarkovChain(chainLength, k, pSame, pObs);
        %     Ate{i} = 0*Ate{i};
    end
    
    plot(Y);
    % A = 0*A;
    
    %% generate overcomplete representation of ground truth
    
    labels = zeros(chainLength*k + nnz(A)*k^2, 1);
    for i = 1:chainLength
        labels(localIndex(i, Y(i), chainLength)) = 1;
    end
    
    [I, J] = find(A);
    for i = 1:nnz(A)
        labels(pairwiseIndex(i, Y(I(i)), Y(J(i)), chainLength, k)) = 1;
    end
    
    
    %% construct structural constraints
    
    
    [Aeq, beq, featureMap] = edge_marginals(X', A, k);
    
    S.Aeq = Aeq;
    S.beq = beq;
    S.A = [];
    S.b = [];
    S.lb = zeros(chainLength*k + nnz(A)*k^2, 1);
    S.ub = [];
    S.x0 = [];
    
    for i = 1:numTest
        [AeqTe, beqTe, featureMapTe{i}] = edge_marginals(Xte{i}', Ate{i}, k);
        %     assert(all(AeqTe(:) == S.Aeq(:)));
    end
    
    %%
    
    for cIndex = 1:length(Cvec)
        for type = 1:3
            
            C = Cvec(cIndex);
            
            if type == 1
                w = vanillaM3N(featureMap, labels, scope, S, C);
                kappa = 0;
            elseif type == 2
                w = learnCRF(featureMap, labels, chainLength*k, S, C);
                kappa = 0;
            else
                [w, kappa] = jointLearnEnt(featureMap, labels, scope, S, C);
                kappa = 0;
            end
            
            y = dualInference(w, featureMap, kappa, S);
            pred = predictMax(y(1:k*chainLength), chainLength, k);
            
            trainError(type, cIndex, run) = nnz(pred ~= Y) / chainLength;
            
            for i = 1:numTest
                y = dualInference(w, featureMapTe{i}, kappa, S);
                pred = predictMax(y(1:k*chainLength), chainLength, k);
                testError(type, cIndex, i, run) = nnz(pred ~= Yte{i}) / chainLength;
            end
            meanTestError(type, cIndex, run) = mean(testError(type, cIndex, :, run));
            varTestError(type, cIndex, run) = var(testError(type, cIndex, :, run));
            
            savedW{cIndex, type, run} = w;
            savedKappa(cIndex, type, run) = kappa;
        end
    end
    
    [Xpred, ~] = find(X');
    baseError(run) = nnz(Xpred~=Y) / length(Y);
    for i = 1:numTest
        [Xpred, ~] = find(Xte{i}');
        baseErrorTe(run, i) = nnz(Xpred~=Yte{i}) / chainLength;
    end
    
end

%%

subplot(311);
semilogx(Cvec, mean(baseError,2) * ones(size(Cvec)), '--k');
hold on;
semilogx(Cvec, mean(trainError, 3), 'x-');
hold off;
axis([min(Cvec), max(Cvec), 0, 1])
title(sprintf('pObs = %f, pSame = %f', pObs, pSame));
ylabel('Training error', 'FontSize', 14);
xlabel('C', 'FontSize', 14);
set(gca, 'FontSize', 14);
legend('Local error', 'M3N', 'CRF', 'CSM3N');

subplot(312);
semilogx(Cvec, mean(baseErrorTe(:)) * ones(size(Cvec)), '--k');
hold on;
semilogx(Cvec, mean(meanTestError, 3), 'x-');
hold off;
axis([min(Cvec), max(Cvec), 0, 1])
ylabel('Avg. testing error', 'FontSize', 14);
xlabel('C', 'FontSize', 14);
set(gca, 'FontSize', 14);

subplot(313);
semilogx(Cvec, mean(varTestError, 3), 'x-');
xlabel('C', 'FontSize', 14);
ylabel('Testing variance', 'FontSize', 14);
set(gca, 'FontSize', 14);

%%
% 
% loglog(Cvec, savedKappa(:,2,5))
% ylabel('kappa');
% xlabel('C');
% 
% %%
% 
% norms = zeros(length(Cvec),1);
% 
% for run = 5:5%1:totalRuns
%     for i = 1:length(Cvec)
%         norms(i) = norm(savedW{i, 1, run});
%     end
%     loglog(Cvec, norms.^2);
%     ylabel('||w||^2');
%     xlabel('C');
%     pause;
% end
% 
% 
