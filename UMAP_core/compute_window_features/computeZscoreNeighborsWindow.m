function [ zscore_neighborsW ] = computeZscore_neighborsWindow(params,x)
%This function calculate the Window feature of score neighbors for vector x. 
 
%Window=2*Radius+1;
 
%For mean Window features
%COL1: original x 
%COL2: offset= -1, window radius= windowRadius
%COL3: offset= 0, window radius= windowRadius
%COL4: offset=1, window radius= windowRadius
 
ROW=size(x,1);
zscore_neighborsW =[];
for windowRadius=params.window.window_radius(2:end)
    zscore_neighborsW_temp=[];
   
    if windowRadius==0
       zscore_neighborsW_temp=x;
    else 
       zscore_neighborsW_temp=[];
       zscore_neighborsW_temp1=[];
       zscore_neighborsW_temp2=[];
       zscore_neighborsW_temp3=[];
       
            if isempty(find(params.window.window_offset==-1))==1
                zscore_neighborsW_temp1=[];
            else 
                for i=1:ROW
                    if i< 2*windowRadius+1
                         zscore_neighborsW_temp1(i)=NaN;
                    else
                         zscore_neighborsW_temp1(i)=(x(i)-nanmean(x((i-2*1) : i)))/nanstd(x((i-2*1) : i));

                    end
                end
            end    
          
            if isempty(find(params.window.window_offset==0))==1
                zscore_neighborsW_temp2=[];
            else     
                for i=1:ROW
                     if (i<windowRadius+1) | (ROW-i<windowRadius) 
                     zscore_neighborsW_temp2(i)=NaN;
                     else
                     zscore_neighborsW_temp2(i)= (x(i)-nanmean(x((i-1):(i+1))))/nanstd(x((i-1):(i+1)))  ;

                     end
                end
            end
            
            if isempty(find(params.window.window_offset==1))==1
                zscore_neighborsW_temp3=[]; 
            else
                for i=1:ROW
                    if ROW-i<2*windowRadius
                         zscore_neighborsW_temp3(i)=NaN;
                    else
                     zscore_neighborsW_temp3(i)= (x(i)-nanmean(x(i:(i+2*1))))/ nanstd(x(i:(i+2*1)));

                    end
                end
            end
    zscore_neighborsW_temp=[zscore_neighborsW_temp1',zscore_neighborsW_temp2',zscore_neighborsW_temp3'];            
    end
    zscore_neighborsW=[zscore_neighborsW,zscore_neighborsW_temp]; 
end
   
 
 
 
 

