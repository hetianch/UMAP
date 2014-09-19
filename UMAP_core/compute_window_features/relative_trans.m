function [ relative_perframe ] = relative_trans( perframe )
%This function is used to relative transform perframe features:dived
%perframe feature of that fly by 80th percentile over entire trajectory.
relative_perframe={};
for i = 1:size(perframe,2)
noninf_perframe=perframe{1,i}(~isinf(perframe{1,i}));
relative_perframe{1,i}=perframe{1,i}./prctile(noninf_perframe,80);
end

