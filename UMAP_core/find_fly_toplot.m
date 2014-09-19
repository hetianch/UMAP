function [ fly_to_plot_array] = find_fly_toplot(diagnosiscandidate,trx)
%This function is used to find flies to plot
% The plot is the prediction of JAABA and NAFF
% It retures the idx of fly to plot in the diagnosiscandidate file.
% Note the real targetNumber is diagnosiscandidate(fly_to_plot_idx).targetNumber

flynumber = length(diagnosiscandidate);
fly_to_plot_array=[];

for i = 1:flynumber
    discrepent_length=size(diagnosiscandidate(i).frame,1);
    trajectory_length=trx(diagnosiscandidate(i).targetNumber).nframes;
    if  discrepent_length/trajectory_length > 0.2 && trajectory_length > 20
       fly_to_plot_array=[fly_to_plot_array;i];
    end
    
    
end

