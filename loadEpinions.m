%% Load data

epinions = load('epinions/epinions-small.txt');
n = max(max(epinions(:,1:2)));
gt = sparse(epinions(:,1),epinions(:,2),2*epinions(:,3)-1,n,n);

% load 'epinions/epinions-big.txt'
% epinions_big(:,1:2) = epinions_big(:,1:2) + 1;
% gt = spconvert(epinions_big);

%% Fix disagreements

% Make disagreements negative
gt(abs(triu(gt-gt'))==2) = -1;
% Make unreciprocated positives positive
gt(triu(gt==1 & gt'==0)) = 1;
gt(triu(gt==0 & gt'==1)) = 1;
% Make unreciprocated negatives negative
gt(triu(gt==-1 & gt'==0)) = -1;
gt(triu(gt==0 & gt'==-1)) = -1;
% Upper-triangulize
gt = triu(gt);

%% Cleanup the workspace (why not!)
clear epinions;
