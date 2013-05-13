

schools = {'cornell', 'texas', 'washington', 'wisconsin'};

allLabels = {'course', 'faculty', 'student', 'research.project', 'other'};


clear label school;

I = [];
J = [];

pageFile = fopen('WebKB/webkb_old/pages.data', 'r');

tokens = textscan(pageFile, '%f\t%s\t%s\t', 'CollectOutput');

id = tokens{1};
[~, label(id)] = ismember(tokens{2}, allLabels);
[~, school(id)] = ismember(tokens{3}, schools);

fclose(pageFile);

%%
% 
% clear words;
% 
% dictFile = fopen('WebKB/webkb_old/words.data', 'r');
% 
% wordIds = [];
% words = {};
% 
% while ~feof(pageFile)
%     line = fgetl(pageFile);
%     tokens = regexp(line,'\t','split');
% 
%     id = str2double(tokens{4});
%     wordIds(end+1) = id;
%     words{end+1} = tokens{2};
% end
% fclose(dictFile);
% 
% [allWords, ~, inds] = unique(words);
% 
% % this is a stupid way to do this
% word2index = zeros(max(wordIds),1);
% for i = 1:length(words)
%     word2index(wordIds(i)) = inds(i);
% end
% 


%%

wordFile = fopen('WebKB/webkb_old/wa.data', 'r');

tokens = textscan(wordFile, '%f\t%f\t%f\t%s\t%s', 'CollectOutput');

I = tokens{2};
J = tokens{3};

X = sparse(I,J,ones(size(I))) > 0;

fclose(wordFile);

%%

linkFile = fopen('WebKB/webkb_old/links.data', 'r');
tokens = textscan(linkFile, '%f\t%f\t%f\t%s', 'CollectOutput');
fclose(linkFile);

I = tokens{2};
J = tokens{3};
[~, Sc] = ismember(tokens{4}, schools);
A = sparse(I,J,ones(size(I)));

%% 

counter = sparse(label(I), label(J), ones(size(I)));


%% split up schools

clear Y cites words;

for i = 1:length(schools)
    inds = school == i;
    
    Y{i} = label(inds);
    
    cites{i} = A(inds,inds);
    words{i} = X(inds,:);
end


