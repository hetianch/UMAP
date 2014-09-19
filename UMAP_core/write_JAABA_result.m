function [ JAABA_result_matrix,originalID ] = write_JAABA_result( scoredgoodTrx )
%This function is used to write JAABA_result into JAABA_result_matrix and
%originalID
%JAABA_result_matrix:
%frame
%multifly

%originalID:
%originalID
%originalID
%originalID

flyID={};
JAABA_result_matrix=[];
flynumber= size(scoredgoodTrx,2);
for i=1:flynumber
    frame_size=size(scoredgoodTrx(i).frame,1);
    flyID_temp=repmat(scoredgoodTrx(i).originalIdx,[1,frame_size] );
    flyID=[flyID, flyID_temp];
    JAABA_result_matrix_temp=[scoredgoodTrx(i).frame,scoredgoodTrx(i).multiFly'];
    JAABA_result_matrix=[JAABA_result_matrix;JAABA_result_matrix_temp];
end
flyID=flyID';
%delete the "" mark of originalID
originalID=strrep(flyID,'"','');