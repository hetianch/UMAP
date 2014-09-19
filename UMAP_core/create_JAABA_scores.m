function out = create_JAABA_scores(python_scores,trx)
%This function is used to convert scores from python package to JAABA
%scores format
%% JAABA?s score format is very 'qi pa'. the scores doesn't start from the first frame of fly
%% but all start from frame1. If the fly doesn't appear at frame 1 then the scores will be filled by 0
%% for example: 
%% tEnd:217	3800	814	4158	19825	2986
%% tStart:1	1	1	5	6	6
%% scores dim:217	3800	814	4158	19825	2986 
allScores=[];
scores={};
timestamp= 7.356129763385634e+05;
tStart=[];
tEnd=[];

flynumber= size(trx,2);
for i = 1: flynumber
    tStart=[tStart,trx(i).firstframe];
    tEnd=[tEnd,trx(i).endframe];
    scores{i}=[zeros(1,trx(i).firstframe-1),(python_scores{i}-0.5)*10];
    postprocessed{i}= scores{i}>0;
end

allScores.scores=scores;
allScores.tStart=tStart;
allScores.tEnd=tEnd;
allScores.postprocessed=postprocessed;

%%% The following is to hack round!!!!!!!
%%% I have to fix this ASAP!!!!
for i = 1:flynumber
    if length(allScores.scores{i})~= trx(i).endframe
        allScores.scores{i}=zeros(1,trx(i).endframe);
    end
end

save JAABAscores allScores timestamp;
end
