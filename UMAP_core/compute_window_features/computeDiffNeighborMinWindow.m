function [ diff_neighbor_minW ] = computeDiffNeighborMinWindow(params,x)
%This function calculate Difference between the current frame's per-frame 
%feature and the minimum of the per-frame feature in the window.

ROW=size(x,1);
diff_neighbor_minW =[];
for windowRadius=params.window.window_radius(2:end)
    diff_neighbor_minW_temp=[];
   
    if windowRadius==0
       diff_neighbor_minW_temp=x;
    else 
       diff_neighbor_minW_temp=[];
       diff_neighbor_minW_temp1=[];
       diff_neighbor_minW_temp2=[];
       diff_neighbor_minW_temp3=[];
       
            if isempty(find(params.window.window_offset==-1))==1
                diff_neighbor_minW_temp1=[];
            else 
                for i=1:ROW
                    if i< 2*windowRadius+1
                         diff_neighbor_minW_temp1(i)=NaN;
                    else
                         diff_neighbor_minW_temp1(i)=x(i)-nanmin(x((i-2*windowRadius) : i));

                    end
                end
            end    
          
            if isempty(find(params.window.window_offset==0))==1
                diff_neighbor_minW_temp2=[];
            else     
                for i=1:ROW
                     if (i<windowRadius+1) | (ROW-i<windowRadius) 
                     diff_neighbor_minW_temp2(i)=NaN;
                     else
                     diff_neighbor_minW_temp2(i)= x(i)-nanmin(x((i-windowRadius):(i+windowRadius)));
                     end
                end
            end
            
            if isempty(find(params.window.window_offset==1))==1
                diff_neighbor_minW_temp3=[]; 
            else
                for i=1:ROW
                    if ROW-i<2*windowRadius
                         diff_neighbor_minW_temp3(i)=NaN;
                    else
                     diff_neighbor_minW_temp3(i)= x(i)-nanmin(x(i:(i+2*windowRadius)));
                    end
                end
            end
    diff_neighbor_minW_temp=[diff_neighbor_minW_temp1',diff_neighbor_minW_temp2',diff_neighbor_minW_temp3'];            
    end
    diff_neighbor_minW=[diff_neighbor_minW,diff_neighbor_minW_temp]; 
end
   
