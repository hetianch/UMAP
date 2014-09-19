function [ goodtrx ] = selectgoodtrx(trx)
%delete flies <10 frame
%set flies with nFlies>1 to missing
goodtrx=trx;
iteri=size(trx,2);
deletelist=[];


for i=1:iteri
    %delete flies < 10 frame
    if size(trx(i).frame,1)<10
       deletelist=[deletelist;i];
    end
end

goodtrx(deletelist')=[];
iteri2=size(goodtrx,2);

for i=1:iteri2
    
    iterj=size(goodtrx(i).frame);
       
        for j=1:iterj
            if  goodtrx(i).nFlies(j)>1
                goodtrx(i).blobX(j)=NaN;
                goodtrx(i).blobY(j)=NaN;
                goodtrx(i).blobArea(j)=NaN;
                goodtrx(i).blobAngle(j)=NaN;
                goodtrx(i).blobA(j)=NaN;
                goodtrx(i).blobB(j)=NaN;
                goodtrx(i).blobdeltaX(j)=NaN;
                goodtrx(i).blobdeltaY(j)=NaN;
                
            end
        end      

                


end

