function storage = learningCallback(x, obj, storage, featureMap, labels)

if isempty(storage)
    storage.kappaVec = [];
    storage.y0 = labels;
    storage.xSaved = zeros(length(x), 0);
end

iter = length(obj);


interval = 10;
if mod(iter-1, interval)==0
    d = size(featureMap,1);
    
    storage.xSaved(:,end+1) = x;
    storage.kappaVec(end+1) = x(d+1);
    
    subplot(411);
    plot(x(1:d), 'x');
    title('W');
    subplot(412);
    plot(1:interval:iter, storage.kappaVec);
    title('kappa');
    xlabel('iter');
    subplot(413);
    plot(x(d+2:end), 'x');
    title('y');
    subplot(414);
    plot(obj);
    title('objective');
    drawnow;
    if iter > 1
        fprintf('Iteration %d, change in obj %d, norm of change in x %d\n', ...
            iter, obj(end) - obj(end-1), norm(storage.xSaved(:,end) - storage.xSaved(:,end-1)));
    end
end
