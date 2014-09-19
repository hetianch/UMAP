%% attach scores to goodtrx
function [scoredgoodTrx] = attachScores_to_goodtrx(goodtrx, allScores)
%this funciton is used to attach postprocessing scores to trajectory
%multi_probability is the probability of having more than one fly in a blob.
%multi_fly is the index of multi fly =1 means more than on fly

% So we need to convert JAABA socres (jaaba scores= 10*(probability-0.5)) back
% to probaiblity first

nflies= size(goodtrx,2);
scoredgoodTrx=goodtrx;

for i=1:nflies
    for j= goodtrx(i).frame(1):goodtrx(i).frame(end)
    multi_probability= allScores.scores{i}(j)/10+0.5;
    scoredgoodTrx(i).multiFly(j-goodtrx(i).frame(1)+1)= allScores.postprocessed{i}(j);
    scoredgoodTrx(i).multiprob(j-goodtrx(i).frame(1)+1)=multi_probability;
    end
end


