function [ meanW ] = computeMeanWindow(params,x)
%This function calculate the Window feature of mean for vector x. 

%Window=2*Radius+1;

%For mean Window features
%COL1: original x 
%COL2: offset= -1, window radius= windowRadius
%COL3: offset= 0, window radius= windowRadius
%COL4: offset=1, window radius= windowRadius

ROW=size(x,1);
meanW =[];
for windowRadius=params.window.window_radius
    meanW_temp=[];
   
    if windowRadius==0
       meanW_temp=x;
    else 
       meanW_temp=[];
       meanW_temp1=[];
       meanW_temp2=[];
       meanW_temp3=[];
       
            if isempty(find(params.window.window_offset==-1))==1
                meanW_temp1=[];
            else 
                for i=1:ROW
                    if i< 2*windowRadius+1
                         meanW_temp1(i)=NaN;
                    else
                         meanW_temp1(i)=nanmean(x((i-2*windowRadius) : i));
                    end
                end
            end    
          
            if isempty(find(params.window.window_offset==0))==1
                meanW_temp2=[];
            else     
                for i=1:ROW
                     if (i<windowRadius+1) | (ROW-i<windowRadius) 
                     meanW_temp2(i)=NaN;
                     else
                     meanW_temp2(i)= nanmean(x((i-windowRadius):(i+windowRadius)));
                     end
                end
            end
            
            if isempty(find(params.window.window_offset==1))==1
                meanW_temp3=[]; 
            else
                for i=1:ROW
                    if ROW-i<2*windowRadius
                         meanW_temp3(i)=NaN;
                    else
                     meanW_temp3(i)= nanmean(x(i:(i+2*windowRadius)));
                    end
                end
            end
    meanW_temp=[meanW_temp1',meanW_temp2',meanW_temp3'];            
    end
    meanW=[meanW,meanW_temp]; 
end
   




