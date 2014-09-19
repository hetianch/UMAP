function [TABU,gSVM] = create_flynumber_perframe(goodtrx,scores_cell_array,delete_fly)
%This function is used to create perframe feature: nFlies by TABU and gSVM 
% 
TABU={};
flynumber=size(goodtrx,2);
for i = 1: flynumber
    TABU{i}=goodtrx(i).nFlies';
end
gSVM={};
for i = 1:flynumber
    if  isempty(find(delete_fly==i))
        gSVM{i}=scores_cell_array{i};
    else
        frame_number=length(goodtrx(i).flyFrame);
        gSVM{i}=zeros(1,frame_number);
    end
end

    
end

