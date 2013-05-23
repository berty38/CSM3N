function [X,Y,A,pSame] = genMarkovChain(n, k, pSameMin, pSameMax, pObs)

assert(pSameMin >= 0)
assert(pSameMax <= 1)
assert(pSameMax >= pSameMin)

X = zeros(n,k);
Y = zeros(n,1);
A = sparse(1:(n-1), 2:n, ones(n-1,1), n, n);

% Init class dist/CDF to uniform
classProbs = ones(k,1)/k;
classCDF = cumsum(classProbs);

% Draw hidden global variable to determine hidden state mixing
pSame = pSameMin + rand * (pSameMax - pSameMin);

% Generate first node
Y(1) = randi(k);
obs = rand < pObs;
if obs
	X(1,Y(1)) = 1;
else
	X(1,randi(k)) = 1;
end

% Generate rest of chain
for i = 2:n
	% Hidden state
	same = rand < pSame;
	if same
		Y(i) = Y(i-1);
	else
		Y(i) = find(rand < classCDF, 1, 'first');
	end
	
	% Observation
	obs = rand < pObs;
	if obs
		X(i,Y(i)) = 1;
	else
		X(i,randi(k)) = 1;
	end
end
