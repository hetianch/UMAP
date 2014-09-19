function proportion = compute_Y_B_proportion( trx )
%This funciton is used to compute the proportion of 'Y' and 'B' in
%blobColor
flynumber= size(trx,2);
num=[];
frame=[];

for i = 1:flynumber
   Y_num_temp=sum( ismember(trx(i).color,'Y'));
   B_num_temp=sum( ismember(trx(i).color,'B'));
   frame_temp=trx(i).nframes;
   num_temp= Y_num_temp+B_num_temp;
  
   num=[num,num_temp];
   frame=[frame,frame_temp];
end

certain_color=sum(num);
total_frame=sum(frame);
proportion= certain_color/total_frame;
