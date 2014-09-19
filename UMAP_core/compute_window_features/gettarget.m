function [window_target] = gettarget(labels)
%This function is used to get labels of video from .jab file of JAABA
% window target is a structure which include flies you labeld on the video
% and their corresponding frams and label names (not flyframe but frame)
%window_target.fly is flyID
%window_target.lables is a matrix:
%frame label
% 100    1
% 101    1
% 102    0

%label "having behavior" is 1
%label "None" is 0
% To hack round: we look after string "None" from labels.names if it is
% empty, means "having behavior"-->1; if it's not empty, means "None"-->0


window_target=[];
empty_cells=cellfun(@isempty,labels.names);
validcells=find(empty_cells==0);
nflies=size(validcells,2);

for i = 1: nflies
    valid_idx=validcells(i);
    window_target(i).fly=labels.flies(valid_idx);
    label_matrix=[];
    for j = 1:size(labels.t0s{valid_idx},2)
        start_frame=labels.t0s{valid_idx}(j);
        end_frame=labels.t1s{valid_idx}(j)-1;
        frame_label=cellfun(@isempty, strfind(labels.names{valid_idx}(j),'None'));
        length=end_frame-start_frame+1;
        
        label_matrix_temp=[];
        label_matrix_temp(:,1)=(start_frame:1:end_frame)';
        label_matrix_temp(:,2)=ones(length,1)*frame_label;
        label_matrix=[label_matrix;label_matrix_temp];
    end
     window_target(i).labels=label_matrix;

end

% check label balance
behavior_frames=[];
none_frames=[];
for i = 1:size(window_target,2)

    behavior_frames_temp=[];
    none_frames_temp=[];
   
    behavior_frames_temp = size(find(window_target(i).labels(:,2)==1),1);
    none_frames_temp=size(find(window_target(i).labels(:,2)==0),1);
    behavior_frames=[behavior_frames,behavior_frames_temp];
    none_frames=[none_frames,none_frames_temp];
end
fprintf('behavior labeled: %d frames\n None labeled: %d frames\n',sum(behavior_frames),sum(none_frames));




