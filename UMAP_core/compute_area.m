function [areaTrx,non_blobArea_single] = compute_area(scoredTrx )
%This function is used to compute the average blobArea of a fly between two
%"multi-fly" status. If the interval  
%first find frames bouts to calculate blobArea
% non_blobArea_single is a single frame between two multi-fly frame. Since it has only
%one frame we cannot calculate blobArea out of it'
areaTrx=scoredTrx;
flynumber=size(scoredTrx,2);

for i =1:flynumber
   
    
    multiIdx=find(scoredTrx(i).multiFly);
    % scoredTrx(i).multiFly=[0,0,1,1,0,0,1]
    % multiIdx=[3,4,7]
   
    %initialize blobArea to zeros
    blobArea=zeros(1,size(scoredTrx(i).frame,1));
    non_blobArea_single=[];
    
    %compute mean blobArea between the interval of "multi-fly"
    for j = 1: length(multiIdx)-1
        
         % if consecutive single frame is less than 2, then we cannot compute the mean blobArea 
               % of this interval. We can give the gender of this frame
               % according to it's neighbour if the neighbouring two
               % genders are the same. If the neighbouring is different, we
               % give the gender of the neighbour with the cloeset size.
               
          
           if multiIdx(j+1)-multiIdx(j)==1
              continue;
          
           elseif multiIdx(j+1)-multiIdx(j)==2
              non_blobArea_single=[non_blobArea_single,multiIdx(j)+1];
           
           %Ex:
              % multiIdx=(1,3,4,5)
              % non_blobArea_single=2
           
           
           else %calculate blobArea average  
             blobArea(multiIdx(j)+1:multiIdx(j+1)-1)=nanmean(blobArea_temp{i}(multiIdx(j)+1:multiIdx(j+1)-1));
             
           end  
         %Then we need to handle that the multiIdx doesn't start from 1 or doesn't end at the end frame:
         if multiIdx(1)==2
             %Ex:  multiFly=(0,1,1,1,0,0)
             %     multiIdx=(2,3,4)
             non_blobArea_single=[1,non_blobArea_single];
         end
         if multiIdx(1)>2
            %Ex: multiFly=(0,0,1,1,0,0)
            %    multiIdx=(3,4)
            blobArea(1:multiIdx(1)-1)=nanmean(blobArea_temp{i}(1:multiIdx(1)-1));
         end
         
         if multiIdx(end)== (size(blobArea,2)-1)
            non_blobArea_single=[non_blobArea_single,size(blobArea,2)];
         end
            
         if size(blobArea,2)-multiIdx(end)>1
             blobArea(multiIdx(end)+1:size(blobArea,2))=nanmean(blobArea_temp{i}(multiIdx(end)+1:size(blobArea,2)));
         end
    end
        
    areaTrx(i).blobArea=blobArea;  
    areaTrx(i).non_blobArea_single=non_blobArea_single;
end

