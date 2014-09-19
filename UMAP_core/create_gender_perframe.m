function [meanarea,speed,speedVariance] = create_gender_perframe(speedTrx)
%This function is used to create perframe feature: nFlies by TABU and gSVM 
% 
meanarea={};
speed={};
speedVariance={};
flynumber=size(speedTrx,2);

for i = 1: flynumber
    speed{i}=speedTrx(i).speed;
end


for i = 1:flynumber
        speedVariance{i}=speedTrx(i).speedVariance;
end


for i = 1:flynumber
        meanarea{i}=speedTrx(i).meanarea;
    
end




end

