initMosek;
initMinFunc;

clear;

chainLength = 100;
pObs = .25;
pSame = .9;

numTest = 100;

k = 3;

%% generate chains

[X,Y,A] = generateMarkovChain(chainLength, k, pSame, pObs);

for i = 1:numTest
    [Xte{i},Yte{i},Ate{i}] = generateMarkovChain(chainLength, k, pSame, pObs);
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
    assert(all(AeqTe(:) == S.Aeq(:)));
end

%%

scope = 1:chainLength*k;

Cvec = 10.^linspace(-4,4,10);

for cIndex = 1:length(Cvec)
    for vanilla = 1:2
        
        C = Cvec(cIndex);
        
        if vanilla == 1
            w = vanillaM3N(featureMap, labels, scope, S, C);
            kappa = 0;
        else
            [w, kappa] = jointLearnEnt(featureMap, labels, scope, S, C);
        end
        
        y = dualInference(w, featureMap, kappa, S);
        pred = predictMax(y(1:k*chainLength), chainLength, k);
        
        trainError(vanilla, cIndex) = nnz(pred ~= Y) / chainLength;
        
        for i = 1:numTest
            y = dualInference(w, featureMapTe{i}, kappa, S);
            pred = predictMax(y(1:k*chainLength), chainLength, k);
            testError(vanilla, cIndex, i) = nnz(pred ~= Yte{i}) / chainLength;
        end
        meanTestError(vanilla, cIndex) = mean(testError(vanilla, cIndex, :));
        varTestError(vanilla, cIndex) = var(testError(vanilla, cIndex, :));
    end
end
%%
trainError
meanTestError
varTestError


%%
[I,J] = find(X);

baseError = nnz(J==Y) / length(Y);
