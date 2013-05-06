function stop = inferenceStat(x,iterationType,i,funEvals,f,t,gtd,g,d,optCond,varargin)

% plot function for inference optimization callback

persistent obj

if i == 1
    obj = [];
end

obj(end+1) = f;

if mod(i-1, 400) == 0
    subplot(211);
    semilogy(obj);
    ylabel('objective');
    xlabel('iteration');
    set(gca, 'FontSize', 16);
    subplot(212);
    inds = max(1, length(obj) - 250):length(obj);
    plot(inds, obj(inds));
    ylabel('objective');
    xlabel('iteration');
    title(sprintf('norm of gradient: %d', norm(g)));
    set(gca, 'FontSize', 16);
    drawnow;
end
stop = false;