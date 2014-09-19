function [ Xrescale_cellarray ] = feature_scale_cellarray( Xoriginal_cellarray )
%This function generalize feature_scale for matrix to cellarray
flynumber = size(Xoriginal_cellarray,2);
Xrescale_cellarray={};


for i=1:flynumber
   Xrescale_cellarray{i}=feature_scale( Xoriginal_cellarray{i});
end


end

