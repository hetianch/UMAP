%function [ closestfly_nose2ell_anglerange,absphidiff,absveldiff,mind,absvelocity_ctr,units] = compute_chase_perframe( trx,anglerange)
function out  = compute_chase_perframe( trx,anglerange)

%This function is used to compute mind, absolute velocity and difference in velocity direction and
%velocity magnitude between current animal and closet animal based on
%nose2ell angle min30 to 30. 

[closestfly_nose2ell_anglerange,mind] = compute_closetfly_n2e_anglerange(trx,anglerange);
%%compute absolute sideway velocity
NFLIES=size(trx,2);
phi = cell(1,NFLIES);
vel = cell(1,NFLIES);
for fly = 1:NFLIES,

    % change in center position
    %dy1 = [trx(fly).y_mm(2)-trx(fly).y_mm(1),(trx(fly).y_mm(3:end)-trx(fly).y_mm(1:end-2))/2,trx(fly).y_mm(end)-trx(fly).y_mm(end-1)];
    %dx1 = [trx(fly).x_mm(2)-trx(fly).x_mm(1),(trx(fly).x_mm(3:end)-trx(fly).x_mm(1:end-2))/2,trx(fly).x_mm(end)-trx(fly).x_mm(end-1)];
    dy1= [diff(trx(fly).y_mm),NaN];
    dx1= [diff(trx(fly).x_mm),NaN];
    % badidx = dy1 == 0 & dx1 == 0;
    phi{fly} = atan2(dy1,dx1);
    vel{fly} = sqrt(dy1.^2+dx1.^2);
   % phi{fly}(badidx) = trx(fly).theta_mm(badidx);
   absvelocity_ctr{fly} = vel{fly};
end
%%compute absphidiff, absveldiff

absphidiff = cell(1,NFLIES);
absveldiff = cell(1,NFLIES); 

for i1 = 1:NFLIES

  fly1 = i1;
  %have to initialize absphidiff and absveldiff!! 
  %have to initialize variable X before use X(idx).
  absphidiff{i1}=nan(1,trx(fly1).nframes);
  absveldiff{i1}=nan(1,trx(fly1).nframes);
  
  % fly closest to fly1 according to type
  closestfly = closestfly_nose2ell_anglerange{fly1};
  
  % velocity direction of fly1
  phi1 = phi{1,fly1};
  vel1= vel{1,fly1};

  % loop over all flies
    for i2 = 1:NFLIES
        fly2 = i2;
        if i1 == i2, continue; end

        % frames where this fly is closest
        idx = find(closestfly(1:end) == fly2);

        %     % don't use the last frame of fly2
        %     off = trx(fly1).firstframe - trx(fly2).firstframe;
        %     idx(idx+off == trx(fly2).nframes) = [];

        if isempty(idx), continue; end

        % orientation of fly2
        off = trx(fly1).firstframe - trx(fly2).firstframe;
        phi2 = phi{fly2}(off+idx);
        vel2 = vel{fly2}(off+idx);

        % absolute difference in orientation
        absphidiff{i1}(idx) = abs(phi2 - phi1(idx));
        absveldiff{i1}(idx) = abs(vel2 - vel1(idx));
    end
end


%% This part is to give -1 to absphidiff, absveldiff and absvelocity_ctr to -1 for mind>20
%% don't know if it will work
for i = 1:NFLIES
    idx= find(mind{i}>20);
    absphidiff{i}(idx)=-1;
    absveldiff{i}(idx)=-1;
   % absvelocity_ctr{i}(idx)=-1;   
end


%%This part is to add abs_closestfly_speed.
for i = 1:NFLIES
    fly1=i;
    %behavior_new(fly1).closetfly_female=-1*ones(size(behavior_new(fly1).frames,2),1);
    frame_with_closestfly= find(~isnan(closestfly_nose2ell_anglerange{fly1}));
    
    fly2_speed=-1*ones(1,trx(fly1).nframes);
    for j= frame_with_closestfly(:)'
 
       % fly2_speed_temp=[];
        fly2=closestfly_nose2ell_anglerange{fly1}(j);
        
        fly1_start_frame=trx(fly1).firstframe;
        fly2_start_frame=trx(fly2).firstframe;
        
        fly1frame= j;

        
        %% frame-fly1_firstframe=fly1frame-1
        %% frame-fly2_firstframe=fly2frame-1
        %% so fly2frame-fly1frame=fly1_firstframe -fly2_firstframe
        fly2frame= trx(fly1).firstframe-trx(fly2).firstframe+fly1frame;
        fly2_speed(j)=absvelocity_ctr{fly2}(fly2frame);
    end    
    abs_closestfly_speed{i}=fly2_speed; 
    
end



units='placehoder';

data=closestfly_nose2ell_anglerange;
save closestfly_nose2ell_anglerange data units;
data=absphidiff;
save absphidiff data units
data=absveldiff;
save absveldiff data units
data=mind;
save mind data units
data=absvelocity_ctr;
save absvelocity_ctr data units
data=abs_closestfly_speed;
save abs_closestfly_speed data units

end


