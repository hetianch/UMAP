function out= match_originalID( goodtrx_new,goodtrx_old )
%This function is used to match originalID in order to know whcih label can
%still be used
for i = 1:size(goodtrx_old,2)
    newID=goodtrx_new(i).originalIdx;
    oldID=goodtrx_old(i).originalIdx;
    if strcmp(newID,oldID) ~=1
        fprintf('found mismatch ID at goodtrx_old fly %d\n',i)
        break
    end


end

