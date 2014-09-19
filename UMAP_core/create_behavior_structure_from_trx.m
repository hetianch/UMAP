function [ behavior] = create_behavior_structure_from_trx( trx_with_sex_chase,closestfly_anglerange,abs_closestfly_speed)
%This function is used to create behavior structure from trx
%trx.multiFly: 1 multi, 0 single
%trx.female_combine: 1 female, 0 male, -1 unknown
%trx.chase: 1 chase, non chase
trx=trx_with_sex_chase;
behavior=[];
flynumber= size(trx,2);
for i = 1: flynumber
    behavior(i).frames=(trx(i).firstframe:1:trx(i).endframe)';
    behavior(i).fly=trx(i).id;
    behavior(i).multiFly=trx(i).multiFly';
    behavior(i).chase=trx(i).chase';
    behavior(i).female=trx(i).female_combine';
    behavior(i).closetfly_min30to30=closestfly_anglerange{i}';
end
%%post process female
%% assign multifly frames to female = -1
for i = 1:flynumber
     multi_idx= find(trx(i).multiFly==1);
     behavior(i).female(multi_idx)=-1;
end

%%post process chase
%assign multifly frames to chase = -1;
%assign closetfly_anglerange=Nan frames to chase=0;
%assign abs_closestfly_speed=0 frames to chase=0;

for i = 1:flynumber
    multi_idx= find(trx(i).multiFly==1);
    closetfly_idx= find(isnan(closestfly_anglerange{i}));
    closetfly_speed_idx= find(abs_closestfly_speed{i}==0);
    
    behavior(i).chase(multi_idx)=-1;
    behavior(i).chase(closetfly_idx)=0;
    behavior(i).chase(closetfly_speed_idx)=0;

   % absvelocity_ctr{i}(idx)=-1;   
end

    