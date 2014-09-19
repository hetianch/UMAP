function [data_nonmissing] = imputenan( data )
%This function is used to impute nan values for time series data.
%The nearby values in a colomns are highly correlated, so we impute the
%missing values by averaging their neighbours. 
%if a colomn contains only nan gives 0 to the entire colomn


data_nonmissing=zeros(size(data,1),size(data,2));
for col=1: size(data,2)
    nonmissing_idx=find(~isnan(data(:,col)));
    gap_idx=[];

    %give original values to nonmissing idx
    data_nonmissing(nonmissing_idx,col)=data(nonmissing_idx,col);

    %give average values of nearest neighbour to missing idx 
    if ~isempty(nonmissing_idx)
         
            % find gap idx, if there's only one nonmissing idx, gap_idx is
            % empty
            for i = 1: length(nonmissing_idx)-1
                if nonmissing_idx(i+1)~=nonmissing_idx(i)+1
                gap_idx=[gap_idx;nonmissing_idx(i); nonmissing_idx(i+1)];
                end  

            end
                if nonmissing_idx(1) ~=1 %handel start from NaN
                    data_nonmissing(1:nonmissing_idx(1)-1,col)=data(nonmissing_idx(1),col)*ones(nonmissing_idx(1)-1,1);
                end

                if nonmissing_idx(end)~= size(data,1)  % handel end with NaN
                   data_nonmissing (nonmissing_idx(end)+1:size(data,1),col)=data(nonmissing_idx(end),col) *ones(size(data,1)-nonmissing_idx(end) ,1);
                end   


            %now let's handel gap case: 1,2,3,4,NaN,NaN,NaN,3,4,5,6
            for k = 1: length(gap_idx)-1
               data_nonmissing(gap_idx(k)+1 : gap_idx(k+1)-1,col)= (data(gap_idx(k),col)+data(gap_idx(k+1),col))/2*ones(gap_idx(k+1)-1-gap_idx(k),1);
            end
    else
        break;
    end
end