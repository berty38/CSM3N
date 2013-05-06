function storage = plotObj(x, obj, storage)

if mod(length(obj)-1, 100) == 0
    fprintf('Iteration %d, obj = %f\n', length(obj), obj(end));
    subplot(211);
    plot(x, 'x');
    title('x');
    subplot(212);
    plot(obj);
    title('Objective');
    drawnow;
end