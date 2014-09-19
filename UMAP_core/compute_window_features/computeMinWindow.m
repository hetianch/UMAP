function [ minW ] = computeMinWindow(params,x)
%This function calculate the Window feature of min for vector x. 

%Window=2*Radius+1;

%For mean Window features
%COL1: original x 
%COL2: offset= -1, window radius= windowRadius
%COL3: offset= 0, window radius= windowRadius
%COL4: offset=1, window radius= windowRadius

ROW=size(x,1);
minW =[];
for windowRadius=params.window.window_radius(2:end)
    minW_temp=[];
   
    if windowRadius==0
       minW_temp=x;
    else 
       minW_temp=[];
       minW_temp1=[];
       minW_temp2=[];
       minW_temp3=[];
       
            if isempty(find(params.window.window_offset==-1))==1
                minW_temp1=[];
            else 
                for i=1:ROW
                    if i< 2*windowRadius+1
                         minW_temp1(i)=NaN;
                    else
                         minW_temp1(i)=nanmin(x((i-2*windowRadius) : i));
                    end
                end
            end    
          
            if isempty(find(params.window.window_offset==0))==1
                minW_temp2=[];
            else     
                for i=1:ROW
                     if (i<windowRadius+1) | (ROW-i<windowRadius) 
                     minW_temp2(i)=NaN;
                     else
                     minW_temp2(i)= nanmin(x((i-windowRadius):(i+windowRadius)));
                     end
                end
            end
            
            if isempty(find(params.window.window_offset==1))==1
                minW_temp3=[]; 
            else
                for i=1:ROW
                    if ROW-i<2*windowRadius
                         minW_temp3(i)=NaN;
                    else
                     minW_temp3(i)= nanmin(x(i:(i+2*windowRadius)));
                    end
                end
            end
    minW_temp=[minW_temp1',minW_temp2',minW_temp3'];            
    end
    minW=[minW,minW_temp]; 
end
   




