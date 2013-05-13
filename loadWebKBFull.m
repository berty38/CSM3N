

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

clear words;

dictFile = fopen('WebKB/webkb_old/words.uniq.data', 'r');

tokens = textscan(dictFile, '%f\t%s\t%s\t%f');

fclose(dictFile);

word_ids = tokens{1};

schoolDicts = {'wo_cornell', 'wo_texas', 'wo_washington', 'wo_wisconsin'};

[allWords, ~, wordMap] = unique(tokens{2});

[~, school_id] = ismember(tokens{3}, schoolDicts);

for i = 1:length(schools)
    inds = school_id == i;
    
    dicts(word_ids(inds),i) = wordMap(inds);
end



%%

wordFile = fopen('WebKB/webkb_old/wa.data', 'r');

tokens = textscan(wordFile, '%f\t%f\t%f\t%s\t%s', 'CollectOutput');

I = tokens{2};
J = tokens{3};

[~, school_id] = ismember(tokens{5}, schoolDicts);

fixed = false(size(J));

for i = 1:length(schools)
    inds = school_id == i;
    J(inds) = dicts(J(inds), i);
    fixed(inds) = true;
end

X = sparse(I,J,ones(size(I))) > 0;

fclose(wordFile);

%% print some random documents

counts = sum(X,1);

[~,inds] = sort(counts, 'descend');

fprintf('most commonly used words:\n');
allWords(inds(1:20))

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


