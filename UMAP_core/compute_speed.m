function [speedTrx,non_speed_single] = compute_speed(scoredTrx )
%This function is used to compute the average speed and area of a fly between two
%"multi-fly" status.  
% scoredTrx is scored goodtrx
%first find frames bouts to calculate speed
% non_speed_single is a single frame between two multi-fly frame. Since it has only
%one frame we cannot calculate speed out of it'
speedTrx=scoredTrx;
flynumber=size(scoredTrx,2);


speed_temp={};
 %compute speed temp
 %Cannot compute the speed of last frame, so:
    for s1=1:flynumber
       
        speed_temp_s1(size(scoredTrx(s1).frame,1))=nan;
    
        for k=1:size(scoredTrx(s1).frame,1)-1
        speed_temp_s1(k)=sqrt((scoredTrx(s1).blobX(k+1)-scoredTrx(s1).blobX(k))^2+(scoredTrx(s1).blobY(k+1)-scoredTrx(s1).blobY(k))^2);
        end
    speed_temp{s1}=speed_temp_s1;
    speed_temp_s1=[];    
    end


for i =1:flynumber
   
    
    multiIdx=find(scoredTrx(i).multiFly);
    % scoredTrx(i).multiFly=[0,0,1,1,0,0,1]
    % multiIdx=[3,4,7]
   
    %initialize speed to zeros
    speed=zeros(1,size(scoredTrx(i).frame,1));
    speedVariance=zeros(1,size(scoredTrx(i).frame,1));
    meanarea=scoredTrx(i).blobArea';
    non_speed_single=[];
    
    %compute mean speed between the interval of "multi-fly"
    for j = 1: length(multiIdx)-1
        
         % if consecutive single frame is less than 2, then we cannot compute the mean speed 
               % of this interval. We can give the gender of this frame
               % according to it's neighbour if the neighbouring two
               % genders are the same. If the neighbouring is different, we
               % give the gender of the neighbour with the cloeset size.
               
          
           if multiIdx(j+1)-multiIdx(j)==1
              continue;
          
           elseif multiIdx(j+1)-multiIdx(j)==2
              non_speed_single=[non_speed_single,multiIdx(j)+1];
           
           %Ex:
              % multiIdx=(1,3,4,5)
              % non_speed_single=2
           
           
           else %calculate speed average  
             speed(multiIdx(j)+1:multiIdx(j+1)-1)=nanmean(speed_temp{i}(multiIdx(j)+1:multiIdx(j+1)-1));
             speedVariance(multiIdx(j)+1:multiIdx(j+1)-1)=nanvar(speed_temp{i}(multiIdx(j)+1:multiIdx(j+1)-1));
             meanarea(multiIdx(j)+1:multiIdx(j+1)-1)=nanmean(scoredTrx(i).blobArea(multiIdx(j)+1:multiIdx(j+1)-1));
           end  
         %Then we need to handle that the multiIdx doesn't start from 1 or doesn't end at the end frame:
         if multiIdx(1)==2
             %Ex:  multiFly=(0,1,1,1,0,0)
             %     multiIdx=(2,3,4)
             non_speed_single=[1,non_speed_single];
         end
         if multiIdx(1)>2
            %Ex: multiFly=(0,0,1,1,0,0)
            %    multiIdx=(3,4)
            speed(1:multiIdx(1)-1)=nanmean(speed_temp{i}(1:multiIdx(1)-1));
            speedVariance(1:multiIdx(1)-1)=nanvar(speed_temp{i}(1:multiIdx(1)-1));
            meanarea(1:multiIdx(1)-1)=nanmean(scoredTrx(i).blobArea(1:multiIdx(1)-1));
         end
         
         if multiIdx(end)== (size(speed,2)-1)
            non_speed_single=[non_speed_single,size(speed,2)];
         end
            
         if size(speed,2)-multiIdx(end)>1
             speed(multiIdx(end)+1:size(speed,2))=nanmean(speed_temp{i}(multiIdx(end)+1:size(speed,2)));
             speedVariance(multiIdx(end)+1:size(speed,2))=nanvar(speed_temp{i}(multiIdx(end)+1:size(speed,2)));
             meanarea(multiIdx(end)+1:size(speed,2))=nanmean(scoredTrx(i).blobArea(multiIdx(end)+1:size(speed,2)));
         end
    end
        
    speedTrx(i).speed=speed;
    speedTrx(i).speedVariance=speedVariance;
    speedTrx(i).meanarea=meanarea;
    speedTrx(i).non_speed_single=non_speed_single;
end

