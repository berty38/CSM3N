function kappa = findKappa(Y, featureMap, labels, w, xi)

vals = zeros(size(Y,2),1);
ytyt = labels' * labels;

for i = 1:size(Y,2)
    y = Y(:,i);
    
    yy = y' * y;
    
    if ytyt > yy
        vals(i) = 2 * (xi - sum(abs(y - labels)) - w' * featureMap * (y - labels)) /...
            (ytyt - yy);
    else
        vals(i) = inf;
    end
end

kappa = min(vals);

