function [ multiclassifier_frame ] = find_multiclassifer_frame( goodtrx )
%This function is used to find frame candidates for multiclass classifier
flynumber=size(goodtrx,2);
multiclassifier_frame={};
for n=1:flynumber
    nFlies_cat=unique(goodtrx(n).nFlies);
    frame_include=[];
    frame_include_temp=[];
    frame_include_temp_exclude_first_20=[];
    exclude_first_20_idx=[];
    for i=1:length(nFlies_cat);
        frame_all=find(goodtrx(n).nFlies==nFlies_cat(i));
        include_idx=(1:ceil(length(frame_all)/100):length(frame_all));
        frame_include_temp=frame_all(include_idx);
        exclude_first_20_idx=find( frame_include_temp>20);
        frame_include_temp_exclude_first_20= frame_include_temp(exclude_first_20_idx);
        frame_include=[frame_include;frame_include_temp_exclude_first_20];
    end
    %%%%multiclassifier_frame{n}=frame_include+goodtrx(n).frame(1)-1;NONON
    %%%%it is more convenient to use flyFrame, not frame.
    multiclassifier_frame{n}=frame_include;
    
end
        

end

