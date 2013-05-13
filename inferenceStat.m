function stop = inferenceStat(x,iterationType,i,funEvals,f,t,gtd,g,d,optCond,varargin)

% plot function for inference optimization callback

persistent obj;
persistent prevX;
persistent changeX;

if i == 0
    obj = [];
    prevX = 0;
    changeX = [];
end

obj(end+1) = f;
changeX(end+1) = norm(x - prevX);
prevX = x;

interval = 100;
window = 500;

if mod(i-1, interval) == 0
    subplot(311);
    if any(obj <= 0)
        plot(obj);
    else
        semilogy(obj);
    end
    ylabel('objective');
    xlabel('iteration');
    set(gca, 'FontSize', 16);
    subplot(312);
    inds = max(1, length(obj) - window):length(obj);
    plot(inds, obj(inds));
    ylabel('objective');
    xlabel(sprintf('last %d iterations', window));
    title(sprintf('norm of gradient: %d', norm(g)));
    set(gca, 'FontSize', 16);
    
    subplot(313);
    semilogy(changeX);
    ylabel('Change in x');
    xlabel('Iteration');
    
    title(sprintf('Optimality condition %d', optCond));
    
    drawnow;
end
stop = false;