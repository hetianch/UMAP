function out = create_JAABAscores_fromB(goodtrx,trx)
%This function is used to convert scores from Brad's prediction to JAABA
%scores format
allScores=[];
scores={};
timestamp= 7.356129763385634e+05;
tStart=[];
tEnd=[];
flynumber= size(trx,2);

%brad's prediction nflies=1,2,3,4, to convert it to JAABA scores, we change
%1 to <0 ,and all the others to >0. So(nflies-1.5)*10


for i = 1: flynumber
    tStart=[tStart,trx(i).firstframe];
    tEnd=[tEnd,trx(i).endframe];
    scores{i}=[zeros(1,trx(i).firstframe-1),(goodtrx(i).nFlies'-1.5)*10]; 
    postprocessed{i}= scores{i}>0;
end

allScores.scores=scores;
allScores.tStart=tStart;
allScores.tEnd=tEnd;
allScores.postprocessed=postprocessed;

%%% The following is to hack round!!!!!!!
%%% I have to fix this ASAP!!!!
%for i = 1:flynumber
%    if length(allScores.scores{i})~= trx(i).endframe
%        allScores.scores{i}=zeros(1,trx(i).endframe);
%    end
%end
save JAABAscores_fromB allScores timestamp;
end
