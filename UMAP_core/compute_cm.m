function out = compute_cm( y_true, y_pred )
%This function is used to compute confusion matrix for y_true and y_pred
%this will give the same results as confusionmat(y_true,y_pred)

%NOTE: y_true, and y_pred should be row vector
y_true_cat=unique(y_true);
%ture_pred=[y_true,y_pred];
accuracy= length(find(y_true==y_pred))/length(y_true);
fprintf('The overall accuracy is : %f\n', accuracy);

    for i=y_true_cat'
        true_idx=find(y_pred==i & y_true==i);
        wrong_idx=find(y_pred ~=i & y_true==i);
        true_pred_count= length(true_idx);
        true_count=length(find(y_true==i));
        
        fprintf('true:%d pred:%d true_count:%d true_pred_count:%d true_pred_count/true_count %f\n',i,i,true_count,true_pred_count,true_pred_count/true_count);

        wrong_pred=y_pred(wrong_idx);
        wrong_pred_cat=unique(wrong_pred);
            for j = wrong_pred_cat'
                wrong_count=length(find( wrong_pred==j));
                fprintf('wrong_pred: %d count: %d \n',j,wrong_count);
            end
        
    end
end

