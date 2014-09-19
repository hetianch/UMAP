function out = convert_JAABAscores_toprob(allScores)
%This function is used to convert scores from allScores.scores in to
%probability and generate prob_fromJAABA file.

flynumber =size(allScores.scores,2);
for i = 1: flynumber 
    %divide by scoreNorm(the 80th percentle of abs(trainingData)
    prob_fromJAABA{i}=allScores.scores{i}/allScores.scoreNorm;
    %set values <-1 to -1
    prob_fromJAABA{i}(find(prob_fromJAABA{i}<-1))=-1;
    %set values >1 to 1
    prob_fromJAABA{i}(find(prob_fromJAABA{i}>1))=1;
    %convert from -1 to 1 range to 0 to 1 range: bying adding 1 and
    %dividing by 2
    prob_fromJAABA{i}= (prob_fromJAABA{i}+1)/2; 

end

allScores.scores=prob_fromJAABA;
timestamp=7.356569931771859e+05;
save prob_fromJAABA allScores timestamp