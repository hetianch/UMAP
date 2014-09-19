function [gt_label_suggest] = rb_gt_suggestion( scores_cell_array_with_frame)
%This function is used to randomly sujest frames for gt labeling in a
%balanced manner.

% convert scores_cell_array_with_frame to a matrix
flynumber=size(scores_cell_array_with_frame,2);
scores_with_frame=[];

emptyCells=cellfun(@isempty,scores_cell_array_with_frame);
validCells=find(emptyCells==0);

for i = validCells
    scores_with_frame_temp=[];
    scores_with_frame_temp(:,1)=ones(length(scores_cell_array_with_frame{i}),1)*i;
    scores_with_frame_temp(:,2)=scores_cell_array_with_frame {i}(:,1);
    scores_with_frame_temp(:,3)=scores_cell_array_with_frame {i}(:,2);
    scores_with_frame=[scores_with_frame;scores_with_frame_temp];
end

% find frames of certain cat and randomly suggest gt labeling in a balanced
% manner
label_cat=unique(scores_with_frame(:,3))';
frame_per_bout=50; % for each label_cat 
bout_num=20; %label 20 bouts for each label_cat 
gt_label_suggest_unsort=[];
for k = label_cat
    gt_label_suggest_temp=[];
    idx=find(scores_with_frame(:,3)==k);
    total_bouts=floor(length(idx)/frame_per_bout);
    % random permutation of total bouts and select bout_num bouts from them
    bout_idx=randperm(int64(total_bouts),int64(bout_num));
    bout_matrix=[];
        for j = bout_idx
           bout_idx_in_scores_with_frame=idx(((j-1)*frame_per_bout+1):j*frame_per_bout);
           bout_matrix_temp=[];
           bout_matrix_temp=scores_with_frame(bout_idx_in_scores_with_frame,:);
           bout_matrix=[bout_matrix;bout_matrix_temp];
        end
     
        
     gt_label_suggest_temp=bout_matrix;
     gt_label_suggest_unsort=[gt_label_suggest_unsort;gt_label_suggest_temp];
end
     gt_label_suggest=sortrows( gt_label_suggest_unsort,2);
end

    
    
    




