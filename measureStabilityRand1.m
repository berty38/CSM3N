function stab = measureStabilityRand1(w, X, k, F, S, type, kappa)

% w : weight vector.
% X : d x n matrix, representing n nodes taking d values.
% k : number of labels in the hidden states.
% F : features
% S : constraints structure.
% type : type of inference to use: 0 for dual, 1 for CRF
% kappa : modulus of convexity

% dimensions
[d,n] = size(X);

% initialize stability to 0
stab = 0;
			
% run initial inference
if type == 1
	y_0 = crfInference(w, F, n*k, S);
else
	y_0 = dualInference(w, F, kappa, S);
end

% perturb each local node
for i=1:n
	
	% select an alternate value
	x_i = find(X(:,i));
	if d == 2
		j = abs(x_i-2)+1;
	else
		otherVals = find(~X(:,i));
		j = otherVals(randi(d-1));
	end
	
	% perturb x_i in X
	X(:,i) = zeros(d,1);
	X(j,i) = 1;

	% recompute local features
	localF = localFeatures(X,k);
	[localm,localn] = size(localF);
	F(1:localm,1:localn) = localF;

	% run inference
	if type == 1
		y_1 = crfInference(w, F, n*k, S);
	else
		y_1 = dualInference(w, F, kappa, S);
	end

	% measure 1-norm and store max
	delta = norm(y_0-y_1, 1);
	if stab < delta
		stab = delta;
	end
	
	% replace perturbed value
	X(j,i) = 0;
	X(x_i,i) = 1;
	
end

