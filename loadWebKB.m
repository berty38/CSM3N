clear;

schools = {'cornell', 'texas', 'washington', 'wisconsin'};

allLabels = {};

for i = 1:length(schools)
    pages{i} = {};
    words{i} = [];
    labels{i} = {};
    pageID = 1;
    fin = fopen(sprintf('WebKB/%s.content', schools{i}), 'r');
    
    while ~feof(fin)
        line = fgetl(fin);
        tokens = regexp(line,'\t','split');
        
        pages{i}{pageID} = tokens{1};
        
        words{i}(pageID,:) = cell2mat(tokens(2:end-1))=='1';
        
        labels{i}{pageID} = tokens{end};
        
        pageID = pageID + 1;
    end
    
    allLabels = union(allLabels, labels{i});
    
    fclose(fin);
    
    fin = fopen(sprintf('WebKB/%s.cites', schools{i}), 'r');
    
    Iname = {};
    Jname = {};
    edgeID = 1;
    while ~feof(fin)
        line = fgetl(fin);
        tokens = regexp(line,' ','split');
        
        Iname{edgeID} = tokens{1};
        Jname{edgeID} = tokens{2};
        edgeID = edgeID + 1;
    end
    
    fclose(fin);
    
    [~,I] = ismember(Iname, pages{i});
    [~,J] = ismember(Jname, pages{i});
    
    cites{i} = sparse(I, J, true(size(I)), length(pages{i}), length(pages{i}));
end

for i = 1:length(schools)
    [~, Y{i}] = ismember(labels{i}, allLabels);
end


