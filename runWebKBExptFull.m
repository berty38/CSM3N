%% Run experiment on webKB using joint minimization


clear;
initMosek;
initMinFunc;

loadWebKBFull;

%%

Cvec = 10.^linspace(-3, 5, 20);

savedW = cell(length(Cvec), 4, 2);
savedKappa = zeros(length(Cvec), 4, 2);
trainError = zeros(length(Cvec), 4, 2);
testError = zeros(length(Cvec), 4, 2);


total = 4 * length(Cvec) * 3 * 2;

count = 1;
totalTimer = tic;

for one_example = 0:1
    
    clear trainError testError savedW savedKappa;
    
    for split = 1:4
        %% set up training and test splits
        if one_example
            train = split;
            test = setdiff(1:4, split);
        else
            train = setdiff(1:4, split);
            test = split;
        end
        
        Xtr = [];
        Ytr = [];
        graphTr = [];
        
        for i = 1:length(train)
            if one_example
                Xtr = [Xtr words{train(i)}'];
            else
                Xtr = [Xtr wordsWo{split}{train(i)}'];
            end
            Ytr = [Ytr; Y{train(i)}(:)];
            tmp = cites{train(i)};
            graphTr = [graphTr, sparse(size(graphTr,1), size(tmp,2));...
                sparse(size(tmp,1), size(graphTr,2)), tmp];
        end
        
        % uncomment to remove relational information
        %  graphTr = 0*graphTr;
        
        Xte = [];
        Yte = [];
        graphTe = [];
        
        for i = 1:length(test)
            if one_example
                Xte = [Xte words{test(i)}'];
            else
                Xte = [Xte wordsWo{split}{test(i)}'];
            end
            Yte = [Yte; Y{test(i)}(:)];
            tmp = cites{test(i)};
            graphTe = [graphTe, sparse(size(graphTe,1), size(tmp,2));...
                sparse(size(tmp,1), size(graphTe,2)), tmp];
        end
        
        k = length(allLabels);
        nTr = size(Xtr, 2);
        nTe = size(Xte, 2);
        d = size(Xtr, 1);
        
        %% compare link label probabilities
        
        [I,J] = find(graphTr);
        counterTr = sparse(Ytr(I), Ytr(J), ones(size(I)));
        [I,J] = find(graphTe);
        counterTe = sparse(Yte(I), Yte(J), ones(size(I)));
        
        subplot(211);
        imagesc(counterTr)
        title('Training');
        subplot(212);
        imagesc(counterTe)
        title('Testing');
        
        
        %% set up local marginal constraints
        
        fprintf('Setting up local marginal polytope\n');
        [Aeq, beq, featureMapTe] = edge_marginals(Xte, graphTe, length(allLabels));
        
        Ste.A = [];
        Ste.b = [];
        Ste.Aeq = Aeq;
        Ste.beq = beq;
        Ste.lb = zeros(nTe * k + nnz(graphTe) * k^2, 1);
        Ste.ub = [];
        Ste.x0 = [];
        
        [Aeq, beq, featureMap] = edge_marginals(Xtr, graphTr, length(allLabels));
        
        S.A = [];
        S.b = [];
        S.Aeq = Aeq;
        S.beq = beq;
        S.lb = zeros(nTr * k + nnz(graphTr) * k^2, 1);
        S.ub = [];
        S.x0 = [];
        
        fprintf('Set up local marginal polytope\n');
        
        clear Aeq beq;
        
        %% expand pairwise potentials
        
        groundTruth = zeros(nTr * k + nnz(graphTr) * k^2, 1);
        
        for i = 1:length(Ytr)
            ind = localIndex(i, Ytr(i), nTr);
            groundTruth(ind) = 1;
        end
        
        %         counter = zeros(k);
        
        [I,J] = find(graphTr);
        for i = 1:nnz(graphTr)
            ind = pairwiseIndex(i, Ytr(I(i)), Ytr(J(i)), nTr, k);
            groundTruth(ind) = 1;
        end
        
        
        groundTruthTe = zeros(nTe * k + nnz(graphTe) * k^2, 1);
        
        for i = 1:length(Yte)
            ind = localIndex(i, Yte(i), nTe);
            groundTruthTe(ind) = 1;
        end
        
        [I,J] = find(graphTe);
        for i = 1:nnz(graphTe)
            ind = pairwiseIndex(i, Yte(I(i)), Yte(J(i)), nTe, k);
            groundTruthTe(ind) = 1;
        end
        
        
        
        
        x0 = [];
        %%
        
        for cIndex = length(Cvec):-1:1
            for type = 1:2
                
                C = Cvec(cIndex);
                
                fprintf('Starting run with C = %f, fold %d\n', C, split);
                
                scope = 1:nTr*k;
                %                                 scope = 1:length(groundTruth);
                
                if type == 1
                    % [w, violation, xi] = vanillaM3N(featureMap, groundTruth, scope, S, C);
                    w = vanillaM3N(featureMap, groundTruth, scope, S, C);
                    kappa = 0;
                elseif type == 2
                    % [w, kappa, y, x0] = jointLearnEnt(featureMap, groundTruth, scope, S, C, x0);
                    [w, kappa, y] = jointLearnEntLog(featureMap, groundTruth, scope, S, C);
                else
                    w = learnCRF(featureMap, groundTruth, nTr*k, S, C);
                    kappa = 1;
                end
                %%
                if type == 3
                    y = crfInference(w, featureMap, nTr*k, S);
                else
                    y = dualInference(w, featureMap, kappa, S);
                end
                
                trainError(cIndex, split, type) =  sum(predictMax(y(1:k*nTr), nTr, k) ~= predictMax(groundTruth(1:k*nTr), nTr, k)) / nTr;
                
                % run on test split
                
                if type == 3
                    y = crfInference(w, featureMap, nTe*k, S);
                else
                    y = dualInference(w, featureMapTe, kappa, Ste);
                end
                
                testError(cIndex, split, type) =  sum(predictMax(y(1:k*nTe), nTe, k) ~= predictMax(groundTruthTe(1:k*nTe), nTe, k)) / nTe;
                
                %%
                fprintf('Training error %f\n', trainError(cIndex, split, type));
                fprintf('Testing error %f\n', testError(cIndex, split, type));
                fprintf('kappa = %d\n', kappa);
                fprintf('0.5 * ||w||^2 = %f\n', 0.5 * w' * w);
                
                savedW{cIndex, split, type}= w;
                savedKappa(cIndex, split, type) = kappa;
                
                
                fprintf('%2.1f percent done (%d of %d). ETA %f minutes for %d runs at %f seconds per run\n', 100 * count / total, ...
                    count, total, (total - count) * (toc(totalTimer) / count) / 60, total - count, toc(totalTimer) / count);
                count = count + 1;
                
            end
            if one_example
%                 save webKBSmallResultsJoint1Ex;
                save webKBFullResultsJoint1Ex;
            else
%                 save webKBSmallResultsJoint3Ex;
                save webKBFullResultsJoint3Ex;
            end
            
        end
    end
end

