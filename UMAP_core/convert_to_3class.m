function [truelabel_3class] = convert_to_3class(truelabel )
%This function is used to change label to 3 category, 0 male, 1 femal,2
%multi

emptyCells=cellfun(@isempty,truelabel);
validCells=find(emptyCells==0);
truelabel_3class=truelabel;

for i = validCells
    idx=find(truelabel_3class{i}(:,3)>1);
    truelabel_3class{i}(idx,3)=2;
end
end

