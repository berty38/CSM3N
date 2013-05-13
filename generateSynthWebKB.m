schools = {'cornell', 'texas', 'washington', 'wisconsin'};

allLabels = {'course', 'faculty', 'student', 'research.project', 'other'};

numWords = 5;

k = length(allLabels);

pagesPerSchool = 25;

wordProb = 0.2;

edgeProb = 3 / pagesPerSchool;

rel = eye(k)+2;

w = [randn(numWords * k, 1); randn(k^2, 1) + rel(:)];



for school = 1:length(schools)
    words{school} = rand(pagesPerSchool, numWords) < wordProb;
    
    cites{school} = rand(pagesPerSchool, pagesPerSchool) < edgeProb;
    
    
    [Aeq, beq, featureMap] = edge_marginals(words{school}', cites{school}, length(allLabels));
    
    S.A = [];
    S.b = [];
    S.Aeq = Aeq;
    S.beq = beq;
    S.lb = zeros(pagesPerSchool * k + nnz(cites{school}) * k^2, 1);
    S.ub = [];
    S.x0 = [];
    
    y = dualInference(w, featureMap, 0, S);
    
    Y{school} = predictMax(y, pagesPerSchool, k);
end


