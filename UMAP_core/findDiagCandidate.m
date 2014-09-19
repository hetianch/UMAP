function [ diagnosiscandidate ] = findDiagCandidate(scoredTrx,goodtrx)
%This function is used to find disagreement of Brad's prediction with JAABA
%prediction
%Bpredict: number of flies in blob
%Jpredict: 1 (1 fly ), 2 (more than 1 fly)
iter1= size(goodtrx,2);
diagnosiscandidate=struct('frame',[],'Bpredict',[],'Jpredict',[],'originalflyID',{});
deletelist=[];

for i=1:iter1
    
diagnosiscandidate(i).frame=[];
diagnosiscandidate(i).Bpredict=[];
diagnosiscandidate(i).Jpredict=[];
diagnosiscandidate(i).originalflyID={};
end


       

for i2=1:iter1
    iter2= size(goodtrx(i2).frame,1);
    diagnosiscandidate(i2).originalflyID=goodtrx(i2).originalIdx;
    diagnosiscandidate(i2).targetNumber=i2;
    for j= 1:iter2
       
        if (scoredTrx(i2).multiFly(j)==0 && goodtrx(i2).nFlies(j)>1) ||...
                (scoredTrx(i2).multiFly(j)==1 && goodtrx(i2).nFlies(j)<2)
           diagnosiscandidate(i2).frame=[diagnosiscandidate(i2).frame;goodtrx(i2).frame(j)];
           diagnosiscandidate(i2).Bpredict=[diagnosiscandidate(i2).Bpredict;goodtrx(i2).nFlies(j)];
           diagnosiscandidate(i2).Jpredict=[diagnosiscandidate(i2).Jpredict;scoredTrx(i2).multiFly(j)+1];
        end
    end

    
end

for i3=1:iter1
    if size(diagnosiscandidate(i3).frame)==0
       deletelist=[deletelist;i3];
    end

end
 diagnosiscandidate(deletelist')=[];
end