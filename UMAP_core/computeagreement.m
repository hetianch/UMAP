%% compute agreement

function [agreement] = computeagreement(diagnosiscandidate,goodtrx)
trxsum=0;
diasum=0;
for i=1:size(goodtrx,2)

trxsum=trxsum+size(goodtrx(i).flyFrame,1);

end


for i=1:size(diagnosiscandidate,2)

diasum=diasum+size(diagnosiscandidate(i).frame,1);

end

agreement= (trxsum-diasum)/trxsum;
end
