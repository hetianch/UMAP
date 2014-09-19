function delete_list = delete_col_onlynan(X)
%This function is used to find colmns with only missing values and delete
%and delete them.
col=size(X,2);
delete_list=[];
for i = 1:col
    nan_row=find(isnan(X(:,1)));
    nan_length=size(nan_row);
    if nan_length==size(X,1)
        delete_list=[delete_list;i];
    end
end

%%% this function has not finished yet! Can only detect but not delete
%%% colomns with only nan