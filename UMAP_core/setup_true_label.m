function [ truelabel_sort ] = setup_true_label( multiclassifier_frame,flytarget,goodtrx )
%This function is used to make a structure to contain true label
truelabel_sort={};
truelabel={};

for i =flytarget
truelabel{i}(:,1)=multiclassifier_frame{i}+goodtrx(i).frame(1)-1;
truelabel{i}(:,2)=goodtrx(i).nFlies(multiclassifier_frame{i});
truelabel{i}(:,3)=zeros(length(truelabel{i}(:,1)),1);

truelabel_sort{i}=sortrows(truelabel{i},1);

end


