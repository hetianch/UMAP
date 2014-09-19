function out = JAABA_postprocessing_normal(goodtrx,allScores)
%This function is used to post process normal JAABA's output:
%1 add scores to goodtrx
%2 calculate agreement with Brad's prediction
%3 find discrepant frames and export to .txt

[scoredTrx] = attachScores(goodtrx, allScores);
[ diagnosiscandidate ] = findDiagCandidate(scoredTrx,goodtrx);
[ diagnosiscandidate_matrix ] = create_diagnosis_candidate_matrix( diagnosiscandidate );
[diagcandidateID]= etrid_from_diagcandidate( diagnosiscandidate)

[agreement] = computeagreement(diagnosiscandidate,goodtrx);
fprintf('the agreement of JAABA and Brad result is %.4f\n',agreement)

dlmcell('flyID.txt',diagcandidateID)
dlmwrite('diagnosiscandidate_matrix.txt',diagnosiscandidate_matrix)
save diagnosiscandidate diagnosiscandidate

%% attach scores to trx
function [scoredTrx] = attachScores(goodtrx, allScores)
%this funciton is used to attach postprocessing scores to trajectory
%multi_probability is the probability of having more than one fly in a blob.
%multi_fly is the index of multi fly =1 means more than on fly

% So we need to convert JAABA socres (jaaba scores= 10*(probability-0.5)) back
% to probaiblity first

nflies= size(goodtrx,2);
scoredTrx=goodtrx;

for i=1:nflies
    for j= goodtrx(i).frame(1):goodtrx(i).frame(end)
    multi_probability= allScores.scores{i}(j)/10+0.5;
    scoredTrx(i).multiFly(j-goodtrx(i).frame(1)+1)= allScores.postprocessed{i}(j);
    scoredTrx(i).multiprob(j-goodtrx(i).frame(1)+1)=multi_probability;
    end
end

end
%% find discrepent frames
function [ diagnosiscandidate ] = findDiagCandidate(scoredTrx,goodtrx)
%This function is used to find disagreement of Brad's prediction with JAABA
%prediction
%Bpredict: number of flies in blob
%Jpredict: probability of having more than one fly
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
           diagnosiscandidate(i2).Jpredict=[diagnosiscandidate(i2).Jpredict;scoredTrx(i2).multiprob(j)];
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
 
                    function diagcandidateID= etrid_from_diagcandidate( diagnosiscandidate)
                    %This function is used to extract diagnosis candidates ID and target number
                    
                    diagcandidateID= {};
                    flynumber=size( diagnosiscandidate,2);
                    
                    for i= 1:flynumber
                    diagcandidateID{i,1}=diagnosiscandidate(i).originalflyID;
                    diagcandidateID{i,2}=diagnosiscandidate(i).targetNumber;
                    end
                    end
                    
                    
                    
                    
                    
                    
                    
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

%%rewrite diagnosis candidate from matlab structure to matrix for inputing
%%into python
function [ diagnosiscandidate_matrix ] = create_diagnosis_candidate_matrix( diagnosiscandidate )
%This function is used to convert diagnosiscandidate to matrix with all
%blank filled with a dummy variable 0 

flynumber = size (diagnosiscandidate,2);
matrix_col = 4*flynumber;

framelength= zeros(1,flynumber);
for i=1: flynumber
    framelength(i)=length(diagnosiscandidate(i).frame);
end

matrix_row = max(framelength);


diagnosiscandidate_matrix = -1*ones(matrix_row,matrix_col);

for i = 1:flynumber
    sub_row= length(diagnosiscandidate(i).frame);
    
    diagnosiscandidate_matrix(1:sub_row,4*i-3)=diagnosiscandidate(i).frame;
    diagnosiscandidate_matrix(1:sub_row,4*i-2)=diagnosiscandidate(i).Bpredict;
    diagnosiscandidate_matrix(1:sub_row,4*i-1)=diagnosiscandidate(i).Jpredict;
end
end
end







