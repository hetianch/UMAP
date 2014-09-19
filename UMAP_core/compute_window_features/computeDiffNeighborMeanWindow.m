function [ diff_neighbor_meanW ] = computeDiffNeighborMeanWindow(params,x)
%This function calculate Difference between the current frame's per-frame 
%feature and the mean of the per-frame feature in the window.

ROW=size(x,1);
diff_neighbor_meanW =[];
for windowRadius=params.window.window_radius(2:end)
    diff_neighbor_meanW_temp=[];
   
    if windowRadius==0
       diff_neighbor_meanW_temp=x;
    else 
       diff_neighbor_meanW_temp=[];
       diff_neighbor_meanW_temp1=[];
       diff_neighbor_meanW_temp2=[];
       diff_neighbor_meanW_temp3=[];
       
            if isempty(find(params.window.window_offset==-1))==1
                diff_neighbor_meanW_temp1=[];
            else 
                for i=1:ROW
                    if i< 2*windowRadius+1
                         diff_neighbor_meanW_temp1(i)=NaN;
                    else
                         diff_neighbor_meanW_temp1(i)=x(i)-nanmean(x((i-2*windowRadius) : i));

                    end
                end
            end    
          
            if isempty(find(params.window.window_offset==0))==1
                diff_neighbor_meanW_temp2=[];
            else     
                for i=1:ROW
                     if (i<windowRadius+1) | (ROW-i<windowRadius) 
                     diff_neighbor_meanW_temp2(i)=NaN;
                     else
                     diff_neighbor_meanW_temp2(i)= x(i)-nanmean(x((i-windowRadius):(i+windowRadius)));
                     end
                end
            end
            
            if isempty(find(params.window.window_offset==1))==1
                diff_neighbor_meanW_temp3=[]; 
            else
                for i=1:ROW
                    if ROW-i<2*windowRadius
                         diff_neighbor_meanW_temp3(i)=NaN;
                    else
                     diff_neighbor_meanW_temp3(i)= x(i)-nanmean(x(i:(i+2*windowRadius)));
                    end
                end
            end
    diff_neighbor_meanW_temp=[diff_neighbor_meanW_temp1',diff_neighbor_meanW_temp2',diff_neighbor_meanW_temp3'];            
    end
    diff_neighbor_meanW=[diff_neighbor_meanW,diff_neighbor_meanW_temp]; 
end
   
