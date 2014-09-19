function [blobColor,true_label] = get_blobColor_for_label( trx,window_target,female_str,male_str )
%This function is used to get blobColor for labed frames
% 1 is female
% 0 is male
flynumber= size(window_target,2);
true_label=[];
blobColor=[];

for i = 1:flynumber
    fly=window_target(i).fly;
    true_label_temp=window_target(i).labels(:,2);
    true_label_frame=window_target(i).labels(:,1);
    true_label_flyframe=true_label_frame-trx(fly).firstframe+1;
    blobColor_str=trx(fly).color(true_label_flyframe)';
    
    % change blobColor_str to values: 1 female, 0 male, -1 unknown
    blobColor_temp= -1 * ones(length(blobColor_str),1);
    female_str_idx=find(ismember(blobColor_str,female_str));
    male_str_idx=find(ismember(blobColor_str,male_str));
    
    blobColor_temp(female_str_idx)=1;
    blobColor_temp(male_str_idx)=0;
    
    true_label=[true_label;true_label_temp];
    blobColor=[blobColor;blobColor_temp];
    
    if length(blobColor_temp)~=length(true_label_temp)
        fprintf('ERROR:length of blobColor and true_label mismatch for fly %d\n',fly);
    end
end
