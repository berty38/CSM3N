function y = overcomplete2label(v, classes)

% Converts an overcomplete representation to a vector of labels.
% v - (nk x 1) vector of assignments
% classes - (k x 1) vector of classes
% y - (n x 1) vector of labels

k = length(classes);
n = length(v) / k;
y = zeros(n,1);
i = 0;
for c=1:k:length(v)
	i = i + 1;
	l = find(v(c:c+k-1));
	y(i) = classes(l);
end


