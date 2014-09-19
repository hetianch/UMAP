function [closestfly_nose2ell_anglerange,mind] = compute_closetfly_n2e_anglerange(trx,anglerange)
%function [d_matrix] = compute_closetfly_n2e_anglerange(trx,anglerange)
%This function is used to compute closet fly based on nose to ell within
%anglerange
nflies=size(trx,2);
flies=(1:1:nflies);
closestfly_nose2ell_anglerange = cell(1,nflies);
mind = cell(1,nflies);
%MAXVALUE=2*trx(1).arena.r; % suppose fly 1 and other flies are all in the same arena
MAXVALUE=21; % give a certain value which is good for training classifier. 

nsamples=1000; %this is a value that correspond to the accuracy of distance between point to ellipse,higher value have higher accuracy;

%anglerange1 = anglerange(1);
%anglerange2= anglerange1(2);


for fly1 = 1:nflies
   
    %initialize distance matrix
	d_matrix = nan(numel(flies),trx(fly1).nframes);
	
    for i2 = 1:numel(flies) 
 

        %initialize 
        d_fly= nan(1,trx(fly1).nframes);
        anglefrom1to2=[];
        idx=[];
        %return if same 
        fly2 = flies(i2);
        if fly1 == fly2
          continue;
        end

        % get start and end frames of overlap
        t0 = max(trx(fly1).firstframe,trx(fly2).firstframe);
        t1 = min(trx(fly1).endframe,trx(fly2).endframe);

        % return if no overlap
        if t1 < t0, 
           continue;
        end

        %get frames of overlap
        start_flyframe1 = t0 + trx(fly1).off;
        end_flyframe1 = t1 + trx(fly1).off;

        start_flyframe2 = t0 + trx(fly2).off;
        end_flyframe2 = t1 + trx(fly2).off;

     %% find frames of fly2 within anglerange of fly1:idx
        anglerange_left=anglerange(1);
        anglerange_right=anglerange(2);
        % position of nose1
        i0=start_flyframe1;
        i1=end_flyframe1;
        j0=start_flyframe2;
        j1=end_flyframe2;

        xnose1 = trx(fly1).x_mm(i0:i1) + 2*trx(fly1).a_mm(i0:i1).*cos(trx(fly1).theta_mm(i0:i1));
        ynose1 = trx(fly1).y_mm(i0:i1) + 2*trx(fly1).a_mm(i0:i1).*sin(trx(fly1).theta_mm(i0:i1));
        theta1 = trx(fly1).theta_mm(i0:i1);

        % position of center2
        x_mm2 = trx(fly2).x_mm(j0:j1);
        y_mm2 = trx(fly2).y_mm(j0:j1);

        dx = x_mm2-xnose1;
        dy = y_mm2-ynose1;

        theta1 = trx(fly1).theta_mm(i0:i1);
        theta2 = atan(dy./dx);

        for k= 1: (i1-i0+1)
%             if theta1(k)<0
%                 if dx(k)>0       
%                    anglefrom1to2(k)=theta2(k)-theta1(k); 
%                 else
%                    anglefrom1to2(k)=-(pi-theta2(k)+theta1(k)); 
%                 end
%             else
%                 if dx(k)>0
%                    anglefrom1to2(k)=-(theta1(k)-theta2(k));
%                 else 
%                    anglefrom1to2(k)=pi+theta2(k)-theta1(k);
%                 end
%             end 
% %             if dx(k)>0
% %                anglefrom1to2(k)=theta2(k)-theta1(k); 
% %             else
% %                anglefrom1to2(k)=pi+theta2(k)-theta1(k); 
% %             end
            if -pi/2<theta1(k) && theta1(k)<0
               if dx(k)>0
                  anglefrom1to2(k)=theta2(k)-theta1(k); 
               else
                   if abs(theta2(k))>abs(theta1(k))
                      anglefrom1to2(k)=-pi+theta2(k)-theta1(k);
                   else
                      anglefrom1to2(k)= pi+theta2(k)-theta1(k);
                   end
               end
           elseif  -pi<theta1(k) && theta1(k)<-pi/2
                if dx(k)<0
                     anglefrom1to2(k)= theta2(k)-theta1(k)-pi; 
                else
                    if abs(theta2(k))+ abs(theta1(k))<pi
                       anglefrom1to2(k)=theta2(k)-theta1(k); 
                    else
                       anglefrom1to2(k)=-2*pi+theta2(k)-theta1(k); 
                    end
                end
             
            elseif theta1(k)>pi/2
                if dx(k) <0
                    anglefrom1to2(k)=pi-theta1(k)+theta2(k); 
                else
                    if abs(theta2(k))+ abs(theta1(k))<pi
                       anglefrom1to2(k)=-theta1(k)+theta2(k);
                    else
                       anglefrom1to2(k)=2*pi+theta2(k)-theta1(k);
                    end
                end
                        
            else
                if dx(k)>0
                       anglefrom1to2(k)= theta2(k)-theta1(k);  
                else
                    if abs(theta2(k))>abs(theta1(k))
                       anglefrom1to2(k)= -pi+theta2(k)-theta1(k); 
                    else
                       anglefrom1to2(k)= pi+theta2(k)-theta1(k); 
                    end
                end
            end


        end
        idx=find((anglefrom1to2>= anglerange_left) & (anglefrom1to2 <= anglerange_right));
      %% compute distance between fly1 and fly2 at frames within anglerange
         a_mm2 = trx(fly2).a_mm(j0:j1);
         b_mm2 = trx(fly2).b_mm(j0:j1);
         theta_mm2 = trx(fly2).theta_mm(j0:j1);

         for i = idx(:)'
         idx_flyframe= i-1+i0;
         d_fly(idx_flyframe) = ellipsedist_hack(x_mm2(i),y_mm2(i),...
         2*a_mm2(i),2*b_mm2(i),theta_mm2(i),...
         xnose1(i),ynose1(i),nsamples);
         end
     
         d_matrix(i2,:)=d_fly;
    end
    [mind{fly1},closesti] = min(d_matrix,[],1);
    closestfly_nose2ell_anglerange{fly1} = flies(closesti);
    closestfly_nose2ell_anglerange{fly1}(isnan(mind{fly1})) = nan;
    mind{fly1}(isnan(mind{fly1})) = MAXVALUE;
end

