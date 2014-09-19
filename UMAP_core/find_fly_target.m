function [fly_target ] = find_fly_target( first_n_fly ,goodtrx,lowerbound)
%This function is used to find fly_target for multiclassifier
% fly target was chosen as (first first_n_fly flies with >lowerbound trajectory lentgh):


trxlength=[];
for i = 1:first_n_fly
    trxlength_temp=length(goodtrx(i).frame);
    trxlength=[trxlength,trxlength_temp];
end

fly_target=find(trxlength>lowerbound);