% global mosek_path
% global minFunc_path
% mosek_path = '/Users/Ben/Code/mosek/6';
% minFunc_path = '/Users/Ben/Code/MATLAB/';
initMosek;
initMinFunc;

clear;

chainLength = 200;
chainLengthTe = 200;
trueW = randn(18,1);
subplot(211);
imagesc(reshape(trueW(1:9), [3 3]));
title('p(x|y)');
subplot(212);
imagesc(reshape(trueW(10:18), [3 3]));
title('p(y^i, y^{i+1})');

numTest = 10;
totalRuns = 20;

k = 3;

scope = 1:chainLength*k;% + (chainLength-1)*k^2;

Cvec = 10.^linspace(-2,6,10);
%Cvec = 10.^linspace(0,5,6);
%Cvec = 10.^linspace(-2,8,10);
% Cvec = [1e-2 1e-1];

total = totalRuns * length(Cvec) * 3;

totalTimer = tic;
count = 0;
%%
for run = 1:totalRuns
    
    %% generate chains
    
    [X,Y,A] = generateMarkovChainW(chainLength, k, trueW);
    
    for i = 1:numTest
        [Xte{i},Yte{i},Ate{i}] = generateMarkovChainW(chainLength, k, trueW);
    end
    
    plot(Y);
    
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
        Ste{i}.Aeq = Aeq;
        Ste{i}.beq = beq;
        Ste{i}.A = [];
        Ste{i}.b = [];
        Ste{i}.lb = zeros(chainLengthTe*k + nnz(Ate{i})*k^2, 1);
        Ste{i}.ub = [];
        Ste{i}.x0 = [];
    end
    
    %% compute true-weight predictions
    
    
    for i = 1:numTest
        y = crfInference(trueW, featureMapTe{i}, chainLength*k, S);
%         y = dualInference(w, featureMap, 0, S);
        pred = predictMax(y(1:k*chainLength), chainLength, k);
        trueWeightErrorTe(i, run) = nnz(pred ~= Yte{i}) / chainLength;
    end
    
    %%
    
    localW = trueW;
    localW(k^2+1:end) = 0;
    
    y = dualInference(localW, featureMap, 0, S);
    pred = predictMax(y(1:k*chainLength), chainLength, k);
    baseError(run) = nnz(pred ~= Y) / chainLength;
    
    for i = 1:numTest
        y = dualInference(localW, featureMapTe{i}, 0, S);
        pred = predictMax(y(1:k*chainLength), chainLength, k);
        baseErrorTe(i, run) = nnz(pred ~= Yte{i}) / chainLength;
    end
    
    %%
    
    for cIndex = length(Cvec):-1:1
        figure(1);
        types = [3 2 1];
        for t=1:length(types)
            type = types(t);
            
            C = Cvec(cIndex);
            
            if type == 1
                w = vanillaM3N(featureMap, labels, scope, S, C);
                kappa = 0;
                y = dualInference(w, featureMap, kappa, S);
            elseif type == 2
                w = learnCRF(featureMap, labels, chainLength*k, S, C);
                kappa = 1;
                y = crfInference(w, featureMap, chainLength*k, S);
                %             elseif type == 3
                %                 [w, kappa] = jointLearnEntLog(featureMap, labels, scope, S, C, [], 1);
                %                 y = dualInference(w, featureMap, kappa, S);
                %             elseif type == 4
                %                 [w, kappa] = jointLearnEntLog(featureMap, labels, scope, S, C, [], 2);
                %                 y = dualInference(w, featureMap, kappa, S);
                %             elseif type == 5
                %                 [w, kappa] = jointLearnEntLog(featureMap, labels, scope, S, C, [], 3);
                %                 y = dualInference(w, featureMap, kappa, S);
            else
                [w, kappa] = jointLearnEntLog(featureMap, labels, scope, S, C);
                y = dualInference(w, featureMap, kappa, S);
            end
            
            pred = predictMax(y(1:k*chainLength), chainLength, k);
            
            trainError(type, cIndex, run) = nnz(pred ~= Y) / chainLength;
            
            for i = 1:numTest
                if type == 2
                    y = crfInference(w, featureMapTe{i}, chainLength*k, S);
                else
                    y = dualInference(w, featureMapTe{i}, kappa, S);
                end
                pred = predictMax(y(1:k*chainLength), chainLength, k);
                testError(type, cIndex, i, run) = nnz(pred ~= Yte{i}) / chainLength;
            end
            meanTestError(type, cIndex, run) = mean(testError(type, cIndex, :, run));
            varTestError(type, cIndex, run) = var(testError(type, cIndex, :, run));
            
            savedW{cIndex, type, run} = w;
            savedKappa(cIndex, type, run) = kappa;
            
            count = count + 1;
            fprintf('Finished %d of %d, elapsed %f minutes, eta %f\n', count,...
                total, toc(totalTimer)/60, (total - count)*(toc(totalTimer)/count)/60);
            
        end
    end
    
    save markovSynthResultsW;
    
    %%
    figure(2);
    subplot(311);
    semilogx(Cvec, mean(baseError,2) * ones(size(Cvec)), '--ko');
    hold on;
    semilogx(Cvec, mean(trainError, 3), 'x-');
    hold off;
    % axis([min(Cvec), max(Cvec), 0, 1])
%     title(sprintf('pObs = %f, pSame = %f', pObs, pSame_tr));
    ylabel('Training error', 'FontSize', 14);
    xlabel('C', 'FontSize', 14);
    set(gca, 'FontSize', 14);
    legend('Local error', 'M3N', 'CRF', 'CSM3N');
    %     legend('Local error', 'M3N', 'CRF', 'CSM3N k=1', 'CSM3N k=2', 'CSM3N k=3', 'CSM3N');
    
    subplot(312);
    semilogx(Cvec, mean(baseErrorTe(:)) * ones(size(Cvec)), '--ko');
    hold on;
    semilogx(Cvec, mean(meanTestError, 3), 'x-');
    hold off;
    % axis([min(Cvec), max(Cvec), 0, 1])
%     title(sprintf('pObs = %f, pSame = %f', pObs, pSame_te));
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
    
    subplot(411);
    
    loglog(Cvec, savedKappa(:,end,run))
    ylabel('kappa');
    xlabel('C');
    title('kappa for current run of CSM3N');
    
    
    norms = zeros(length(Cvec),1);
    subplot(412);
    for i = 1:length(Cvec)
        norms(i) = norm(savedW{i, 1, run});
    end
    loglog(Cvec, norms.^2);
    ylabel('||w||^2');
    xlabel('C');
    title('norm for M3N');
    
    
    subplot(413);
    for i = 1:length(Cvec)
        norms(i) = norm(savedW{i, 2, run});
    end
    loglog(Cvec, norms.^2);
    ylabel('||w||^2');
    xlabel('C');
    title('norm for CRF');
    
    
    subplot(414);
    for i = 1:length(Cvec)
        norms(i) = norm(savedW{i, 3, run});
    end
    loglog(Cvec, norms.^2);
    ylabel('||w||^2');
    xlabel('C');
    title('norm for CSM3N');
    
end

