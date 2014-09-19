function [ maxW ] = computeMaxWindow(params,x)
%This function calculate the Window feature of max for vector x. 

%Window=2*Radius+1;

%For mean Window features
%COL1: original x 
%COL2: offset= -1, window radius= windowRadius
%COL3: offset= 0, window radius= windowRadius
%COL4: offset=1, window radius= windowRadius

ROW=size(x,1);
maxW =[];
for windowRadius=params.window.window_radius(2:end)
    maxW_temp=[];
   
    if windowRadius==0
       maxW_temp=x;
    else 
       maxW_temp=[];
       maxW_temp1=[];
       maxW_temp2=[];
       maxW_temp3=[];
       
            if isempty(find(params.window.window_offset==-1))==1
                maxW_temp1=[];
            else 
                for i=1:ROW
                    if i< 2*windowRadius+1
                         maxW_temp1(i)=NaN;
                    else
                         maxW_temp1(i)=nanmax(x((i-2*windowRadius) : i));
                    end
                end
            end    
          
            if isempty(find(params.window.window_offset==0))==1
                maxW_temp2=[];
            else     
                for i=1:ROW
                     if (i<windowRadius+1) | (ROW-i<windowRadius) 
                     maxW_temp2(i)=NaN;
                     else
                     maxW_temp2(i)= nanmax(x((i-windowRadius):(i+windowRadius)));
                     end
                end
            end
            
            if isempty(find(params.window.window_offset==1))==1
                maxW_temp3=[]; 
            else
                for i=1:ROW
                    if ROW-i<2*windowRadius
                         maxW_temp3(i)=NaN;
                    else
                     maxW_temp3(i)= nanmax(x(i:(i+2*windowRadius)));
                    end
                end
            end
    maxW_temp=[maxW_temp1',maxW_temp2',maxW_temp3'];            
    end
    maxW=[maxW,maxW_temp]; 
end
   




