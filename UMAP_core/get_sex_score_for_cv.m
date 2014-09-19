function [sex_score,true_label]= get_sex_score_for_cv(sex_scored_trx,window_target,sexIdx)
%This function is just used for get sex_scores of cv labed frames

flynumber= size(window_target,2);
true_label=[];
sex_score=[];

for i = 1:flynumber
    fly=window_target(i).fly;
    true_label_temp=window_target(i).labels(:,2);
    true_label_frame=window_target(i).labels(:,1);
    true_label_flyframe=true_label_frame-sex_scored_trx(fly).firstframe+1;
    sex_score_temp=sex_scored_trx(fly).(sexIdx)(true_label_flyframe)';
    
    true_label=[true_label;true_label_temp];
    sex_score=[sex_score; sex_score_temp];
    
    if length(sex_score_temp)~=length(true_label_temp)
        fprintf('ERROR:length of sex_score and true_label mismatch for fly %d\n',fly);
    end
end

