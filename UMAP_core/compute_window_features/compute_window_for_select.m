% function [ window_matrix,label_vector] = compute_window_for_select( window_target,perframe,trx,params)
% %this function is used to compute window_features for selected fly and
% %frames
% window_for_select=struct();
% nflies=size(window_target,2);
% window_matrix=[];
% label_vector=[];
% %for i = 1:nflies
% for i = 9
%     window_matrix_temp=[];
%     label_vector_temp=[];
%     
%     fly=window_target(i).fly;
%     window_for_select(i).flyID=fly;
%     for j= 1: length(window_target(i).labels)
%   % for j = 14
%        flyframe=window_target(i).labels(j,1)+1-trx(fly).firstframe;
%        window_for_select(i).perframe(j)=perframe{window_target(i).fly}(flyframe);
%     end
%     window_for_select(i).perframe=window_for_select(i).perframe';
%     window_for_select(i).window=wrap_compute_window(params,window_for_select(i).perframe);
%     
%     window_matrix_temp=window_for_select(i).window;
%     window_matrix=[window_matrix;window_matrix_temp];
%     label_vector_temp=window_target(i).labels(:,2);
%     label_vector=[label_vector;label_vector_temp];
% end
% 

function [ select_window,label_vector] = compute_window_for_select(all_window,window_target,trx)
%%all window is window feature for all frames
%%select_window is window features for selected frames (window_target)
select_window=[];
label_vector=[];
%for k= 1: size(window_target,2);
    for k=4
    select_window_temp=[];
    
    label_vector_temp=window_target(k).labels(:,2);
    label_vector=[label_vector;label_vector_temp];
    
    fly=window_target(k).fly;
    select_frames = window_target(k).labels(:,1)+1-trx(fly).firstframe;
    select_window_temp = all_window(fly).window(select_frames,:);
    select_window=[select_window;select_window_temp];
end    