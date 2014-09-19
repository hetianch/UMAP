function out = checkPythonscores_cell_array(scores_cell_array,trx)
%This function is used to check the dimension of Python scores cell array
%size(scores{i},2) should be equal to tEnd
flynumber = size(scores_cell_array,2);
for i =1:flynumber
    nframes=trx(i).endframe-trx(i).firstframe+1;
    if size(scores_cell_array{i},2)~= nframes
       fprintf('find dim mismatch for fly %f\n',i);
    end

end

