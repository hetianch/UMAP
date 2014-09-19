function [ window_matrix,label_vector] = compute_window_features(labels,params,trx,perframe_path)
%This is the main function of compute_window_features package.
% if your per frame is at /Users/071cht/Desktop/chase/sample/perframe/mind.mat
% perframe_path = '/Users/071cht/Desktop/chase/sample/chase/perframe/'
[window_target] = gettarget(labels);
perframes=fieldnames(params.perframe);
window_matrix=[];
%for i = 1: numel(perframes)
 for   i=4
    perframe=perframes{i};
    filename=strcat(perframe_path,perframe,'.mat'); % we put perframe folder under the current directory
    addpath(perframe_path); %% won't read from this perframe folder if not adding to matlab path
    loadperframe=load(filename);
    
    transform=params.perframe.(perframe).trans_types;
    window_matrix_temp=[];
    
    if isempty(find(ismember(transform,'none')))==1
        window_matrix_temp1=[];
    else
        data=loadperframe.data;
        all_window= compute_all_window(trx,data,params);
        [window_matrix_temp1,y_vector] = compute_window_for_select(all_window,window_target,trx);
        clear all_window;
    end
    
    if isempty(find(ismember(transform,'abs')))==1
        window_matrix_temp2=[];
    else
        data=abs_trans(loadperframe.data);
        all_window= compute_all_window(trx,data,params);
        [window_matrix_temp2,y_vector] = compute_window_for_select(all_window,window_target,trx);
         clear all_window;
    end
    
    if isempty(find(ismember(transform,'relative')))==1
        window_matrix_temp3=[];
    else
        data=relative_trans(loadperframe.data);
        all_window= compute_all_window(trx,data,params);
        [window_matrix_temp3,y_vector] = compute_window_for_select(all_window,window_target,trx);
         clear all_window;
    end
    
    window_matrix_temp=[window_matrix_temp1,window_matrix_temp2,window_matrix_temp3];
    window_matrix=[window_matrix,window_matrix_temp];
end

label_vector=y_vector;

