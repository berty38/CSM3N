initMosek;
initMinFunc;

clear;

n = 100;
pObs = .1;
pSame = .9;

k = 3;

[X,Y,A] = generateMarkovChain(n, k, pSame, pObs);
[Xte,Yte,Ate] = generateMarkovChain(n, k, pSame, pObs);

plot(Y);

[Aeq, beq, featureMap] = edge_marginals(X', A, k);

S.Aeq = Aeq;
S.beq = beq;
S.A = [];
S.b = [];
S.lb = zeros(n*k + nnz(A)*k^2, 1);
S.ub = [];
S.x0 = [];

[AeqTe, beqTe, featureMapTe] = edge_marginals(Xte', Ate, k);

assert(all(AeqTe(:) == S.Aeq(:)));


labels = zeros(n*k + nnz(A)*k^2, 1);

for i = 1:n
    labels(localIndex(i, Y(i), n)) = 1;
end

[I, J] = find(A);
for i = 1:nnz(A)
    labels(pairwiseIndex(i, Y(I(i)), Y(J(i)), n, k)) = 1;
end


scope = 1:n*k;


Cvec = 10.^linspace(-4,4,20);

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
        pred = predictMax(y(1:k*n), n, k);
        
        trainError(vanilla, cIndex) = nnz(pred ~= Y) / n;
        
        y = dualInference(w, featureMapTe, kappa, S);
        pred = predictMax(y(1:k*n), n, k);
        testError(vanilla, cIndex) = nnz(pred ~= Yte) / n;
    end
end
%%
trainError

testError
