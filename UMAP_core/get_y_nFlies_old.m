function [ nFlies] = get_y_nFlies( goodtrx,multiclassifier_frame,flytarget)
%This function is used to find the nFlies in goodtrx for the frames for
%certin fly targets


nFlies=[];
for i = flytarget
   
    nFlies_temp=goodtrx(i).nFlies(multiclassifier_frame{i});
    nFlies=[nFlies;nFlies_temp] ;
end    
end

