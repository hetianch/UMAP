function out = frame_diagnosis_trx( trx )
%This function is used to check nframes endframe first frame and dimention
%of other fileds

flynumber = size(trx,2);
for i = 1: flynumber
    nframes_predict=trx(i).endframe-trx(i).firstframe+1;
    x=size(trx(i).x,2);
    y=size(trx(i).y,2);
    theta=size(trx(i).theta,2);
    if  (nframes_predict ~= trx(i).nframes) ||(trx(i).nframes ~=x) ||(trx(i).nframes ~=y)||(trx(i).nframes ~=theta)
       fprintf('find dim mismatch for target %f\n', trx(i).id) ;
    end  

end

