function v = label2overcomplete(y, classes)

% Converts a vector of labels to an overcomplete representation.
% y - (n x 1) vector of labels
% classes - (k x 1) vector of classes
% v - (nk x 1) vector of assignments

n = length(y);
k = length(classes);
v = zeros(n*k,1);
i = 0;
for c=1:k:n
	i = i + 1;
	l = find(ismember(y(i), classes));
	v(c-1+l) = 1;
end
	
