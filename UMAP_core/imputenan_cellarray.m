function [ data_nonmissing_cellarray ] = imputenan_cellarray( data_cellarray)
%Extend imputenan from matrix to cellarray 
flynumber = size(data_cellarray,2);
data_nonmissing_cellarray={};
for i=1:flynumber
    data_nonmissing_cellarray{i}=imputenan(data_cellarray{i});
end

