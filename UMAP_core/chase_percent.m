function out= chase_percent(behavior)
%This function is used to calcualte percent of male-male, male-female
%chase percent

flynumber = size(behavior,2);
frame_total_count=[];
multi_total_count=[];
female_total_count=[];
chase_frame_count=[];
male_chase_male_count=[];
male_chase_female_count=[];
male_chase_both_count=[];

female_chase_female_count=[];
female_chase_male_count=[];

for i = 1: flynumber
    %% calculate total frames numbers
    frame_total_temp=size(behavior(i).frames,1);
    frame_total_count=[frame_total_count,frame_total_temp];
    %% calculate multifly frames
    multi_total_temp=length(find(behavior(i).multiFly==1));
    multi_total_count=[multi_total_count, multi_total_temp];
    %% calculate female frames
    female_total_temp=length(find(behavior(i).female==1));
    female_total_count=[female_total_count, female_total_temp];
    %% calculate chase frames numbers
    chase_temp=size(find(behavior(i).chase==1),1);
%     chase_temp_percent= chase_temp/frame_total_temp;
%     if ~isempty( chase_temp)
%     fprintf('fly %d chase frame /total frame: %6.2f /n',i, chase_temp_percent);
    chase_frame_count=[chase_frame_count,chase_temp];
    %% calculate male chase male frame numbers
    male_chase_male_temp=size(find(behavior(i).chase==1 & behavior(i).female==0 & behavior(i).closetfly_female==0),1);
%     male_chase_male_temp_percent= male_chase_male_temp/frame_total_temp;
%     if ~isempty(male_chase_male_temp)
%     fprintf('fly %d male chase male frame /total frame: %6.2f /n',i, male_chase_male_temp_percent);
    male_chase_male_count=[male_chase_male_count,male_chase_male_temp];
    %% calculate male chase female frame numbers
    male_chase_female_temp=size(find(behavior(i).chase==1 & behavior(i).female==0 & behavior(i).closetfly_female==1),1);
%     male_chase_female_temp_percent= male_chase_female_temp/frame_total_temp;
%     if ~isempty(male_chase_female_temp)
%     fprintf('fly %d male chase female frame /total frame: %6.2f /n',i, male_chase_male_temp_percent);
    male_chase_female_count=[male_chase_female_count,male_chase_female_temp];
    
    %%male chase both
    male_chase_both_temp=size(find(behavior(i).chase==1 & behavior(i).female==0),1);
    male_chase_both_count=[male_chase_both_count, male_chase_both_temp];
    
    %%calculate female chase female frame numbers
    female_chase_female_temp=size(find(behavior(i).chase==1 & behavior(i).female==1 & behavior(i).closetfly_female==1),1);
    female_chase_female_count=[female_chase_female_count,female_chase_female_temp];
    
    %%calculate female chase female frame numbers
    female_chase_male_temp=size(find(behavior(i).chase==1 & behavior(i).female==1 & behavior(i).closetfly_female==0),1);
    female_chase_male_count=[female_chase_male_count,female_chase_male_temp];
end
chase_percent=sum(chase_frame_count)/sum(frame_total_count)*100;
male_chase_male_percent=sum(male_chase_male_count)/sum(frame_total_count) *100;
male_chase_female_percent=sum(male_chase_female_count)/sum(frame_total_count)*100;
male_chase_both_percent=sum(male_chase_both_count)/sum(frame_total_count)*100;

total_chase= sum(male_chase_male_count)+sum(male_chase_female_count)+sum(female_chase_male_count)+sum(female_chase_female_count);

fprintf('total fly frames: %d \n',sum(frame_total_count));
fprintf('multi fly frames: %d, multi/total: %6.4f%% \n',sum(multi_total_count),sum(multi_total_count)/sum(frame_total_count)*100 );
fprintf('female fly frames: %d, female/total: %6.4f%% \n',sum(female_total_count),sum(female_total_count)/sum(frame_total_count)*100);

%fprintf('total chase frames: %d, total chase percent: %6.4f%% \n',sum(chase_frame_count),chase_percent);
%fprintf('male chase others: %d, male chase others percent: %6.4f%% \n',sum(male_chase_both_count),male_chase_both_percent);

fprintf('total chase: %d, total chase/total frame : %6.4f%%\n',total_chase,total_chase/sum(frame_total_count)*100);

fprintf('aggression(male chase male): count:%d, percent:%6.4f%% \n',sum(male_chase_male_count),male_chase_male_percent);
fprintf('courtship (male chase female):count:%d, percent: %6.4f%% \n',sum(male_chase_female_count),male_chase_female_percent);
fprintf('female chase female: count:%d, percent:%6.4f%% \n',sum(female_chase_female_count),sum(female_chase_female_count)/sum(frame_total_count)*100);
fprintf('female chase male: count:%d, percent:%6.4f%% \n',sum(female_chase_male_count),sum(female_chase_female_count)/sum(frame_total_count)*100);



%%% have to comment the behavior percent for each single fly. Because one
%%% flyID may correspond to different flies.