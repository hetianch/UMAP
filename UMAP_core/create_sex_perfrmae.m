function out = create_sex_perfrmae(goodtrx,allScores)
%create sex perframe feature
[scoredTrx] = attachScores_to_goodtrx(goodtrx, allScores);
[speedTrx,non_speed_single] = compute_speed(scoredTrx);

data=[];
units='mm/s';
flynumber = size(speedTrx,2);

for i = 1:flynumber
    data{i}= speedTrx(i).speed;
    nframes=length(speedTrx(i).frame);
    perframe_dim=length(speedTrx(i).speed);
    if nframes ~= perframe_dim
        fprintf('dimension mismach of speed for fly %d\n',i);
    end
end

save speed data units 

data=[];
units='mm2/s2';
for i = 1:flynumber
    data{i}= speedTrx(i).speedVariance;
    
    nframes=length(speedTrx(i).frame);
    perframe_dim=length(speedTrx(i).speedVariance);
    if nframes ~= perframe_dim
        fprintf('dimension mismach of speed variance for fly %d\n',i);
    end
end

save speedVariance data units

data=[];
units='mm2';
for i = 1:flynumber
    data{i}= speedTrx(i).meanarea;
    
    nframes=length(speedTrx(i).frame);
    perframe_dim=length(speedTrx(i).meanarea);
    if nframes ~= perframe_dim
        fprintf('dimension mismach of mean area for fly %d\n',i);
    end
end
save area data units
end

 
  