function [ nflies_frame ] = find_fly( goodtrx, nflies_to_find)
%This function is used to find the fly frame corresponding to nflies in a
%blob.

flynumber=size(goodtrx,2);
nflies_frame={};
nflies_frame_size=[];

for n=1:flynumber  
        nflies_frame_temp=find(goodtrx(n).nFlies==nflies_to_find);
        nflies_frame{n}=nflies_frame_temp+goodtrx(n).frame(1)-1;
        nflies_frame_size=[nflies_frame_size;length( nflies_frame_temp)];
end
        
fprintf('%d frames found\n', sum(nflies_frame_size))

end

