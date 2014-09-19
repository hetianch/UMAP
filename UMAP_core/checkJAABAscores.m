function out = checkJAABAscores(allScores)
%This function is used to check the dimension of JAABA scores
%size(scores{i},2) should be equal to tEnd
flynumber = size(allScores.scores,2);
for i =1:flynumber
    if length(allScores.scores{i})~=allScores.tEnd(i)
       fprintf('find dim mismatch for fly %f\n',i);
    end

end

