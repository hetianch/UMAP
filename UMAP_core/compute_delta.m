%%convert blob angle to JAABA theta
function [goodtrx_delta] = compute_delta(goodtrx)
%This function is used to compute deltaX an deltaY;
goodtrx_delta=goodtrx;
for i=1: size(goodtrx,2);
    %set all deltaX to nan 
    dim=length(goodtrx(i).blobX);
    deltaX=nan(1,dim);
    deltaY=nan(1,dim);
    %set second frame of delta
    if goodtrx(i).nFlies(1)==1 && goodtrx(i).nFlies(2)==1 && abs(goodtrx(i).blobArea(2)-goodtrx(i).blobArea(1))<100
       deltaX(2)=goodtrx(i).blobX(2)-goodtrx(i).blobX(1);
       deltaY(2)=goodtrx(i).blobY(2)-goodtrx(i).blobY(1);
    end
    
    for j = 2:dim
        %only update delta value when certain condition is satisfied!
        if goodtrx(i).nFlies(j)==1 %&& ~isnan(deltaX(j-1))
           if abs(goodtrx(i).blobArea(j)-goodtrx(i).blobArea(j-1))<100 % or framesD>5
            
              deltaX(j)=deltaX(j-1)*0.7+(goodtrx(i).blobX(j)-goodtrx(i).blobX(j-1))*0.3;
           else
              deltaX(j)=deltaX(j-1);
           end
        end
        
    end
    
     for j = 2:dim
        %only update delta value when certain condition is satisfied!
        if goodtrx(i).nFlies(j)==1 && ~isnan(deltaY(j-1))
           if abs(goodtrx(i).blobArea(j)-goodtrx(i).blobArea(j-1))<100 % or framesD>5
            
              deltaY(j)=deltaY(j-1)*0.7+(goodtrx(i).blobY(j)-goodtrx(i).blobY(j-1))*0.3;
           else
              deltaY(j)=deltaY(j-1);
           end
        end
        
     end
    goodtrx_delta(i).blobdeltaX=deltaX';
    goodtrx_delta(i).blobdeltaY=deltaY';
    
end
end
