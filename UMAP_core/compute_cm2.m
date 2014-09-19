function out = compute_cm2(y_true,y_pred)
%this function is used to compute the sensitivity
%,specificity,precision,and accuracy of confusion matrix

% y_true should be binary 0,1
% y_pred can be whatever category

cm=confusionmat(y_true,y_pred);

true_pos=length(find(y_true==1));
true_neg=length(find(y_true==0));

for i = 1:size(cm,1)
    if sum(cm(i,:))==true_pos
       pos_row_idx=i;
    end
    if sum(cm(i,:))==true_neg
       neg_row_idx=i;
    end
end

test_pos= sum(cm(:,pos_row_idx));

TP=cm(  pos_row_idx,pos_row_idx);
TN=cm(  neg_row_idx,neg_row_idx);

ACC= (TP+TN)/(true_pos+true_neg);
SEN= TP/true_pos;
SPE= TN/true_neg;
Precision= TP/test_pos;

fprintf('ACC: %.4f\n',ACC);
fprintf('SEN: %.4f\n',SEN);
fprintf('SPE: %.4f\n',SPE);
fprintf('Prc: %.4f\n',Precision);

    
end

