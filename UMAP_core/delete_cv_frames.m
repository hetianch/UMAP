
function [ scores_cell_array_with_frame ] = delete_cv_frames( scores_cell_array,truelabel,goodtrx,delete_fly )


%This function is used to delete frames in cv set and delete_fly from ground truth

flynumber=size(scores_cell_array,2);
scores_cell_array_with_frame={};
for i = 1: flynumber
   
    scores_cell_array_with_frame{i}(:,1)=(1:1:length(scores_cell_array{i}))+goodtrx(i).frame(1)-1;
    scores_cell_array_with_frame{i}(:,2)=scores_cell_array{i}';
end



%for i = 1: flynumber
%    frame_num=split_idx(2*i)-split_idx(2*i-1)+1;
%    python_scores_matrix_temp(:,1)=ones(frame_num,1)*i;
%    python_scores_matrix_temp(:,2)=(1:1:frame_num)+goodtrx(i).frame(1)-1;
%    python_scores_matrix_temp(:,3)=python_scores(split_idx(2*i-1):split_idx(2*i));
%    
%    python_scores_matrix=[python_scores_matrix;python_scores_matrix_temp];
    
%end


% now let's delete frames in truelabel from python_scores_matrix
emptyCells=cellfun(@isempty,truelabel);
validCells=find(emptyCells==0);
for i = validCells
    delete_list=[];
    for j = 1:size(truelabel{i},1)
        delete_list_frame_temp=(truelabel{i}(j,1):1:truelabel{i}(j,2))';
        delete_list_flyframe_temp=delete_list_frame_temp+1-goodtrx(i).frame(1);
        delete_list=[delete_list;delete_list_flyframe_temp];
    end
    scores_cell_array_with_frame{i}(delete_list,:)=[];
end

%last let's delete flies in deletelist
for k = 1:flynumber
    if ~isempty(find(delete_fly==k))
       scores_cell_array_with_frame{k}=[];
    else
        continue;
    end
end

end

        
 