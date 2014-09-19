function [ behavior_new ] = find_chased_gender( behavior_with_gender,trx )
%This function is used to find the gender of fly that is being chased
%1 female, 0 male, -1 unknown
behavior_new=behavior_with_gender;
flynumber= size(behavior_new,2);
for i = 1: flynumber; 

  
    fly1=i;
    %behavior_new(fly1).closetfly_female=-1*ones(size(behavior_new(fly1).frames,2),1);
    frame_with_closestfly= find(~isnan(behavior_new(i).closetfly_min30to30));
   % frame_with_closestfly= find(behavior_new(i).closetfly_min30to30>=0);
    fly2_female=-1*ones(size(behavior_new(fly1).frames,1),1);
   for j= frame_with_closestfly(:)'
  

       % fly2_female_temp=[];
        fly2=behavior_new(fly1).closetfly_min30to30(j);
        
        fly1_start_frame=trx(fly1).firstframe;
        fly2_start_frame=trx(fly2).firstframe;
        
        fly1frame= j;

        
        %% frame-fly1_firstframe=fly1frame-1
        %% frame-fly2_firstframe=fly2frame-1
        %% so fly2frame-fly1frame=fly1_firstframe -fly2_firstframe
        fly2frame= trx(fly1).firstframe-trx(fly2).firstframe+fly1frame;
        fly2_female(j)=behavior_new(fly2).female(fly2frame);
    end    
    behavior_new(i).closetfly_female=fly2_female;
end

