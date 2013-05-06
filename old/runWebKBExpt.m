clear;
initMosek;

loadWebKB;
% trimWords;

%%

tolerance = 1e-3;

Cvec = 5.^[-2 -1 0 1 2 3];
kappaVec = linspace(0, 4, 41);

total = 4 * length(Cvec) * length(kappaVec);

count = 1;
totalTimer = tic;

for split = 1:4
    test = setdiff(1:4, split);
    train = split;
    
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
    
    %% training
    
    fprintf('Setting up QP relaxations\n');
    [Aeq, beq, featureMapTe] = edge_marginals(Xte, graphTe, length(allLabels));
    
    Ste.A = [];
    Ste.b = [];
    Ste.Aeq = Aeq;
    Ste.beq = beq;
    Ste.lb = zeros(nTe * k + nnz(graphTe) * k^2, 1);
    Ste.ub = [];
    Ste.x0 = [];
    Ste.options = optimset('algorithm', 'active-set', 'display', 'off');
    Ste.options.MSK_IPAR_INTPNT_NUM_THREADS = 4;
    
    [Aeq, beq, featureMap] = edge_marginals(Xtr, graphTr, length(allLabels));
    
    S.A = [];
    S.b = [];
    S.Aeq = Aeq;
    S.beq = beq;
    S.lb = zeros(nTr * k + nnz(graphTr) * k^2, 1);
    S.ub = [];
    S.x0 = [];
    S.options = optimset('algorithm', 'active-set', 'display', 'off');
    S.options.MSK_IPAR_INTPNT_NUM_THREADS = 4;
    
    fprintf('Set up QP relaxations\n');
    
    clear Aeq beq;
    
    %% expand pairwise potentials
    
    groundTruth = zeros(nTr * k + nnz(graphTr) * k^2, 1);
    
    for i = 1:length(Ytr)
        ind = localIndex(i, Ytr(i), nTr);
        groundTruth(ind) = 1;
    end
    
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
    
    
    %%
    
    for cIndex = 1:length(Cvec)
        for kappaIndex = 1:length(kappaVec)
            
            C = Cvec(cIndex);
            
            kappa = kappaVec(kappaIndex);
            
            fprintf('Starting run with C = %f, kappa = %f, fold %d\n', C, kappa, split);
            
            [w, violation, xi] = stableSVMStruct(featureMap, groundTruth, S, C, kappa, tolerance, 1:nTr*k, d, k);
%             [w, violation, xi] = stableSVMStruct(featureMap, groundTruth, S, C, kappa, tolerance, 1:(nTr*k + nnz(graphTr)*k*k), d, k);
            
            %%
            y = inference(w, kappa, featureMap, S);
            
            trainError(cIndex, kappaIndex, split) =  sum(predictMax(y(1:k*nTr), nTr, k) ~= predictMax(groundTruth(1:k*nTr), nTr, k)) / nTr;
            
            % run on test split
            
            y = inference(w, kappa, featureMapTe, Ste);
            
            testError(cIndex, kappaIndex, split) =  sum(predictMax(y(1:k*nTe), nTe, k) ~= predictMax(groundTruthTe(1:k*nTe), nTe, k)) / nTe;
            
            
            fprintf('Training error %f\n', trainError(cIndex, kappaIndex, split));
            fprintf('Testing error %f\n', testError(cIndex, kappaIndex, split));
            fprintf('kappa = %f\n', kappa);
            fprintf('xi = %f\n', xi);
            fprintf('0.5 * ||w||^2 = %f\n', 0.5 * w' * w);
            
            savedW{split}{cIndex}{kappaIndex} = w;
            
            
            fprintf('%d percent done (%d of %d). ETA %f minutes for %d runs at %f seconds per run\n', 100 * count / total, ...
                count, total, (total - count) * (toc(totalTimer) / count) / 60, total - count, toc(totalTimer) / count);
            count = count + 1;
            
            save webKBResultsFine2;
        end
    end
end

