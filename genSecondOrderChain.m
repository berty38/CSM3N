function [X,Y,A] = genSecondOrderChain(n, k, p2Hop, pSame, pObs)

I = 1:n-1;
J = 2:n;
A = sparse(I,J,ones(n-1,1), n, n);

X = zeros(n,k);

Y = zeros(n,1);

classProbs = ones(k,1)/k; % uniform
classCDF = cumsum(classProbs);

% Generate first two nodes
Y(1) = randi(k);
obs = rand < pObs;
if obs
	X(1,Y(1)) = 1;
else
	X(1,randi(k)) = 1;
end
same = rand < pSame(1);
if same
	Y(2) = Y(1);
else
	Y(2) = find(rand < classCDF, 1, 'first');
end
obs = rand < pObs;
if obs
	X(2,Y(2)) = 1;
else
	X(2,randi(k)) = 1;
end

for i=3:n
	
	% Generate hidden state
	hop = (rand < p2Hop) + 1;
	same = rand < pSame(hop);
	if same
		Y(i) = Y(i-hop);
	else
		Y(i) = find(rand < classCDF, 1, 'first');
	end
	
	% Generate observation
	obs = rand < pObs;
	if obs
		X(i,Y(i)) = 1;
	else
		X(i,randi(k)) = 1;
	end
	
end
