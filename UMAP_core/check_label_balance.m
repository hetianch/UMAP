function out = check_label_balance( truelabel )
%this function is used to check number of frames labeled in each category
flynumber = size(truelabel,2);
%only work with non empty cells
emptyCells=cellfun(@isempty,truelabel);
validCells=find(emptyCells==0);

label_cat=[];
    for k = validCells
        label_cat_temp=unique(truelabel{k}(:,3));
        label_cat=[label_cat;label_cat_temp];
    end

    label_cat=unique(label_cat);
    for j=1:length(label_cat);
         labelcount=[];
         labelcount_allfly=[];
        for i = 1:length(validCells)
            count_idx=find(truelabel{validCells(i)}(:,3)==label_cat(j));
            labelend=truelabel{validCells(i)}(:,2);
            labelstart=truelabel{validCells(i)}(:,1);
            labelcount_temp=sum(labelend(count_idx)-labelstart(count_idx)+1);
            labelcount=[labelcount;labelcount_temp];
        end
        labelcount_allfly=sum(labelcount);
        fprintf('label %d frame %d\n',label_cat(j), labelcount_allfly);
    end
end
        
        

