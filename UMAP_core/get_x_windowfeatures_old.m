function [ multiclass_x] = get_x_windowfeatures( whole,flytarget,multiclassifier_frame )
%This function is used to get the windowfeatures of certain frames of
%certain flytargets
multiclass_x=[];

for i =flytarget
    
 multiclass_x_temp=whole{i}(multiclassifier_frame{i},:) ;  
 multiclass_x=[multiclass_x;multiclass_x_temp];
end

