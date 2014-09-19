function [ stdW ] = computeStdWindow(params,x)
%This function calculate the Window feature of max for vector x. 

%Window=2*Radius+1;

%For mean Window features
%COL1: original x 
%COL2: offset= -1, window radius= windowRadius
%COL3: offset= 0, window radius= windowRadius
%COL4: offset=1, window radius= windowRadius

ROW=size(x,1);
stdW =[];
for windowRadius=params.window.window_radius(2:end)
    stdW_temp=[];
   
    if windowRadius==0
       stdW_temp=x;
    else 
       stdW_temp=[];
       stdW_temp1=[];
       stdW_temp2=[];
       stdW_temp3=[];
       
            if isempty(find(params.window.window_offset==-1))==1
                stdW_temp1=[];
            else 
                for i=1:ROW
                    if i< 2*windowRadius+1
                         stdW_temp1(i)=NaN;
                    else
                         stdW_temp1(i)=nanstd(x((i-2*windowRadius) : i));
                    end
                end
            end    
          
            if isempty(find(params.window.window_offset==0))==1
                stdW_temp2=[];
            else     
                for i=1:ROW
                     if (i<windowRadius+1) | (ROW-i<windowRadius) 
                     stdW_temp2(i)=NaN;
                     else
                     stdW_temp2(i)= nanstd(x((i-windowRadius):(i+windowRadius)));
                     end
                end
            end
            
            if isempty(find(params.window.window_offset==1))==1
                stdW_temp3=[]; 
            else
                for i=1:ROW
                    if ROW-i<2*windowRadius
                         stdW_temp3(i)=NaN;
                    else
                     stdW_temp3(i)= nanstd(x(i:(i+2*windowRadius)));
                    end
                end
            end
    stdW_temp=[stdW_temp1',stdW_temp2',stdW_temp3'];            
    end
    stdW=[stdW,stdW_temp]; 
end
   




