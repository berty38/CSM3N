function w = handTune(pObs, pSame, k)

truePObs = pObs + (1/3)*(1-pObs);
truePSame = pSame + (1/3)*(1-pSame);

local = eye(k) * log(truePObs) + (1-eye(k)) * log((1-truePObs)/2);

rel = eye(k) * log(truePSame) + (1-eye(k)) * log((1-truePSame)/2);

w = [local(:); rel(:)];

