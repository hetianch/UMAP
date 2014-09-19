function out = create_JAABAscores_fromP(python_scores,split_idx,trx )
%This function is used to convert scores from python package to JAABA
%scores format

% First read in python_scores and split_idx (created by create_wholeframe_matrix.m) to convert scores from python
% to cell array of scores of each fly.

flynumber = size(split_idx,1)/2;
scores_cell_array={};

for i = 1: flynumber
    
    scores_cell_array{i}=python_scores(split_idx(2*i-1):split_idx(2*i))';
end

%second create JAABA scores from the above cell array

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

for i = 1: flynumber
    tStart=[tStart,trx(i).firstframe];
    tEnd=[tEnd,trx(i).endframe];
    scores{i}=[zeros(1,trx(i).firstframe-1),(scores_cell_array{i}-0.5)*10];
    postprocessed{i}= scores{i}>0;
end

allScores.scores=scores;
allScores.tStart=tStart;
allScores.tEnd=tEnd;
allScores.postprocessed=postprocessed;

%%% The following is to hack round!!!!!!! In some cases the JAABA cannot
%%% output the whole frame window features for some flies with a very very
%%% long tracks (>100,000). Before We figure this out, we can only solve
%%% this problem by setting the scores calculated for these flies to 0, to
%%% make the groud truth mode work!!!
%%% I have to fix this ASAP!!!!

for i = 1:flynumber
    if length(allScores.scores{i})~= trx(i).endframe
        allScores.scores{i}=zeros(1,trx(i).endframe);
    end
end

save JAABAscores_fromP allScores timestamp;
end



