function out= frame_diagnosis( goodtrx )
%thif function is used to find frame which is 1,1,2,3,4,5....instead of
%1,2,3,4,5

flynumber = size (goodtrx,2);
for i =1:flynumber
    framenumber= size(goodtrx(i).frame,1) ;
    for j = 2: framenumber
        if goodtrx(i).frame(j) ~=goodtrx(i).frame(j-1)+1
fprintf('find inconsecutive frame for fly %s at frame %f\n', goodtrx(i).originalIdx{1},goodtrx(i).frame(j))
        end
    end 
end

