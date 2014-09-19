function [ abs_perframe ] = abs_trans( perframe )
%This function is used to relative transform perframe features:dived
%perframe feature of that fly by 80th percentile over entire trajectory.
abs_perframe={};
for i = 1:size(perframe,2)
abs_perframe{1,i}=abs(perframe{1,i});
end

