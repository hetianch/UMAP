function [ TABU ] =get_TABU( gt_label_suggest,goodtrx )
%This function is used to get the corresponding TABU results of the flies
%and frames in gt_label_suggest

TABU=[];
iter=size(gt_label_suggest,1);
for i = 1: iter
TABU_temp=[];
fly=gt_label_suggest(i,1);
frame=gt_label_suggest(i,2);
flyframe=gt_label_suggest(i,2)-goodtrx(fly).frame(1)+1;
TABU_temp=goodtrx(fly).nFlies(flyframe);
TABU=[TABU;TABU_temp];
end

