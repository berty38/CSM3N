function [stab_mu, stab_y] = measureStabilityRand1(w, X, k, F, S, type, kappa)

% w : weight vector.
% X : d x n matrix, representing n nodes taking d values.
% k : number of labels in the hidden states.
% F : features
% S : constraints structure.
% type : type of inference to use: 0 for dual, 1 for CRF
% kappa : modulus of convexity
%
% stab_mu : 1-norm stability of marginals
% stab_y : Hamming stability of decoding

% dimensions
[d,n] = size(X);

% initialize stability to 0
stab_mu = 0;
stab_y = 0;
			
% run initial inference
if type == 1
	y_0 = crfInference(w, F, n*k, S);
else
	y_0 = dualInference(w, F, kappa, S);
end
pred_0 = predictMax(y_0(1:k*n), n, k);

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
	delta = norm(y_0(1:k*n)-y_1(1:k*n), 1);
	if stab_mu < delta
		stab_mu = delta;
	end
	
	% measure Hamming distance of decoding and store max
	pred_1 = predictMax(y_1(1:k*n), n, k);
	delta = nnz(pred_0 ~= pred_1);
	if stab_y < delta
		stab_y = delta;
	end
	
	% replace perturbed value
	X(j,i) = 0;
	X(x_i,i) = 1;
	
end

