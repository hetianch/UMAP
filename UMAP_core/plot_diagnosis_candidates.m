function out = plot_diagnosis_candidates( scores_fromJAABA,diagnosiscandidate,trx,fly_to_plot_array,true_number,goodtrx)
%This function is used to plot the prediction of diagnossi candidate from
%JAABA and NAFF
%for i = fly_to_plot  % looping through array in matlab is easier than in Python!! Python, you should cry!
%But this way is not easy to get index of i so let's use the original
%stupid way.
%fly_to_plot_idx is the idex of fly_to_plot in diagnosiscandidate file.


%%% NOTICE: skew the line a little bit to make them don't overlap

%%% multi: true=1, JAABA_predid= 0.99,N_pred=0.98    
%%% single:true=0, JAABA_predict=0.01, N_pred=0.02
%%% true: 0.97,0.03

%%% JAABA_pred: abs(original value-0.01)
%%% N_pred=abs(oribinal value-0.02)

for j=1:length(fly_to_plot_array)
    fly_to_plot_idx=fly_to_plot_array(j); 
    fly_to_plot=diagnosiscandidate(fly_to_plot_idx).targetNumber;
    
    trajectory_length=trx( fly_to_plot).nframes;
    start_frame=trx( fly_to_plot).firstframe;
    end_frame=trx( fly_to_plot).firstframe+trajectory_length-1;
    
    frame_to_plot_boost=(start_frame:1: end_frame); %this is the entire trajectory
   % trajectory of that fly, then I decided to only plot the discrepent
   % frames
    frame_to_plot_x=(diagnosiscandidate(fly_to_plot_idx).frame)';
   
  % JAABA_probability=prob_fromJAABA.scores{fly_to_plot}(start_frame:end_frame);
    JAABA_scores=scores_fromJAABA.postprocessed{fly_to_plot}(start_frame:end_frame);
    
    %plot only discrepent frames
    %JAABA_scores_fordiscrepentframe=scores_fromJAABA.postprocessed{fly_to_plot}(diagnosiscandidate(fly_to_plot_idx).frame);
    
    
    %plot entire trajectory
    JAABA_pred_ori=JAABA_scores;
    JAABA_pred=abs(JAABA_pred_ori-0.01);
    % so the frame_to_plot_JAABA is the same as frame_to_plot_NAFF
    
    frame_to_plot_JAABA= frame_to_plot_boost;
    
    %To plot only discrepent frames for NAFF
    %NAFF_diagnosis_candidates_pred_ori= (diagnosiscandidate(fly_to_plot_idx).Bpredict>1)'; % Bpredict>1, NAFF_pred=1, multifly; nFlies=1, NAFF_pred=0, single fly
    %NAFF_diagnosis_candidates_pred=abs( NAFF_diagnosis_candidates_pred_ori-0.01);
    
    %To plot the entire prediction of NAFF
    NAFF_pred_ori=(goodtrx(fly_to_plot).nFlies>1)';
    NAFF_pred=abs( NAFF_pred_ori-0.02);
    
    frame_to_plot_NAFF=  frame_to_plot_boost; % this is the entire trajectory
    
    frame_to_plot_true= frame_to_plot_x;
    true_to_plot_ori=true_number{j};
true_to_plot=abs(true_to_plot_ori-0.03);

    
    %make plots
fh=figure(j);
set(fh,'color','white') % sets the color to white
%set(gca,'Box','off');
    set(gca,'YTick',[0,1])
    plot(frame_to_plot_JAABA,JAABA_pred,'b-.',frame_to_plot_NAFF,NAFF_pred,'r--o', frame_to_plot_true,true_to_plot,'k-','LineWidth',1.2,'MarkerSize', 5)
    hleg=legend('gentle boost','NAFF','truth');
    set(hleg,'Location','EastOutside')
    hold on
    title( sprintf('the truth and multi fly classification by gentle boost and NAFF for fly %i',fly_to_plot),'FontSize',13)
    xlabel('frame','FontSize',13,'FontWeight','normal')
    ylabel('multi fly index for a blob','FontSize',13,'Rotation',90,'FontWeight','normal')
    saveas(fh,strcat('disagreement_plot',num2str(fly_to_plot)),'epsc')
    
end

