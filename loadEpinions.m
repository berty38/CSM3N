%% Load data
epinions = load('epinions/epinions-small.txt');
n = max(max(epinions(:,1:2)));
gt = sparse(epinions(:,1),epinions(:,2),2*epinions(:,3)-1,n,n);
% load 'epinions/epinions-big.txt'
% epinions_big(:,1:2) = epinions_big(:,1:2) + 1;
% gt = spconvert(epinions_big);

%% Snowball sample a reasonably sized connected component

