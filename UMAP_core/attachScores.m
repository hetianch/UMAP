function [scoredTrx, scores] = attachScores(trx, allScores,score_name)
%this funciton is used to attach postprocessing scores to trajectory
%multiFly=1 means this fly is in a blob with other flies at this frame. 0
%means unknown 
nflies= size(trx,2);
scoredTrx=trx;
scores = {};

for i=1:nflies
    for j= trx(i).firstframe:trx(i).endframe
    scoredTrx(i).(score_name)(j-trx(i).firstframe+1)= allScores.postprocessed{i}(j);
    scores{i}(j-trx(i).firstframe+1)= allScores.scores{i}(j);
    end
end


