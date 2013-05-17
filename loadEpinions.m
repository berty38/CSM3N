%% Load data
% load 'epinions/epinions-big.txt'
% epinions_big(:,1:2) = epinions_big(:,1:2) + 1;
% gt = spconvert(epinions_big);
load 'epinions/epinions-small.txt'
epinions_small(:,3) = 2*epinions_small(:,3) - 1;
gt = spconvert(epinions_small);

%% Snowball sample a reasonably sized connected component

