function out = check_fly_cat(goodtrx )
%This function is used to check the category of nFlies in this goodtrx
%structure

flynumber = size(goodtrx,2);
label_cat=[];
for k = 1:flynumber
        label_cat_temp=unique(goodtrx(k).nFlies);
        label_cat=[label_cat;label_cat_temp];
end
    
label_cat_total=unique(label_cat);

fprintf('there are %d category of labels\n',length(label_cat_total));

   
    for j=1:length(label_cat_total);
         labelcount=[];
         labelcount_allfly=[];
        for i = 1:flynumber
            count_idx=find(goodtrx(i).nFlies==label_cat_total(j));
            labelcount_temp=length(count_idx);
            labelcount=[labelcount;labelcount_temp];
        end
        labelcount_allfly=sum(labelcount);
        fprintf('label %d have %d frames\n',label_cat_total(j), labelcount_allfly);
    end
end
        

