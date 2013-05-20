%% Run experiment on webKB using joint minimization


clear;
initMosek;
initMinFunc;

generateSynthWebKB;

%%

Cvec = 10.^linspace(-3, 0, 10);

tolerance = 1e-3;
total = 4 * length(Cvec) * 2 * 2;

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
            Xtr = [Xtr words{train(i)}'];
            Ytr = [Ytr; Y{train(i)}(:)];
            tmp = cites{train(i)};
            graphTr = [graphTr, sparse(size(graphTr,1), size(tmp,2));...
                sparse(size(tmp,1), size(graphTr,2)), tmp];
        end
        
        graphTr = graphTr;
        
        Xte = [];
        Yte = [];
        graphTe = [];
        
        for i = 1:length(test)
            Xte = [Xte words{test(i)}'];
            Yte = [Yte; Y{test(i)}(:)];
            tmp = cites{test(i)};
            graphTe = [graphTe, sparse(size(graphTe,1), size(tmp,2));...
                sparse(size(tmp,1), size(graphTe,2)), tmp];
        end
        
        
        k = length(allLabels);
        nTr = size(Xtr, 2);
        nTe = size(Xte, 2);
        d = size(Xtr, 1);
        
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
%             
%             counter(Ytr(I(i)), Ytr(J(i))) = ...
%                 counter(Ytr(I(i)), Ytr(J(i))) + 1; 
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
            for vanilla = 1:1
                
                C = Cvec(cIndex);
                
                fprintf('Starting run with C = %f, fold %d\n', C, split);
                
                scope = 1:nTr*k;

                if vanilla == 1
                   % [w, violation, xi] = vanillaM3N(featureMap, groundTruth, scope, S, C);
                     w = vanillaM3N(featureMap, groundTruth, scope, S, C);
                    kappa = 0;
                else
%                     [w, kappa, y, x0] = jointLearnEnt(featureMap, groundTruth, scope, S, C, x0);
                    [w, kappa, y] = jointLearnEnt(featureMap, groundTruth, scope, S, C);
                end
                %%
                y = dualInference(w, featureMap, kappa, S);
                
                trainError(cIndex, split, vanilla) =  sum(predictMax(y(1:k*nTr), nTr, k) ~= predictMax(groundTruth(1:k*nTr), nTr, k)) / nTr;
                
                % run on test split
                
                y = dualInference(w, featureMapTe, kappa, Ste);
                
                testError(cIndex, split, vanilla) =  sum(predictMax(y(1:k*nTe), nTe, k) ~= predictMax(groundTruthTe(1:k*nTe), nTe, k)) / nTe;
                
                %%
                fprintf('Training error %f\n', trainError(cIndex, split, vanilla));
                fprintf('Testing error %f\n', testError(cIndex, split, vanilla));
                fprintf('kappa = %d\n', kappa);
                fprintf('0.5 * ||w||^2 = %f\n', 0.5 * w' * w);
                
                savedW{vanilla}{split}{cIndex}= w;
                savedKappa{vanilla}(split, cIndex) = kappa;
                
                
                fprintf('%2.1f percent done (%d of %d). ETA %f minutes for %d runs at %f seconds per run\n', 100 * count / total, ...
                    count, total, (total - count) * (toc(totalTimer) / count) / 60, total - count, toc(totalTimer) / count);
                count = count + 1;
                
                if one_example
                    save webKBSynthResultsJoint1Ex;
                else
                    save webKBSynthResultsJoint3Ex;
                end
            end
        end
    end
end

