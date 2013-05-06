%% quick script to trim words from WebKB to top 100 used, for debugging only

counts = zeros(size(words{1},2),1);

for i = 1:length(words)
    counts = counts + sum(words{i},1)';
end

% trim to most common 100 words

[~,inds] = sort(counts, 'descend');

for i = 1:length(words)
    words{i} = words{i}(:,inds(1:100));
end
