%function out = create_JAABA_trajectory(flyID,flymatrix,blobColor,x,y,d)
    %This function is used to create JAABA input file : registered_trx.mat from
    %original .csv trajectory file

    [roughtrx] = createTrajectory(flyID,flymatrix,blobColor);
    [goodtrx] = selectgoodtrx(roughtrx);
    [thetaPtoJpi] = convertThetaPtoJ(goodtrx);
    [trx] = create_gpJAABAformat(goodtrx,thetaPtoJpi,x,y,d);
    timestamps=(1:1:max(flymatrix(:,1))) ./30;

    save registered_trx timestamps trx;
    save goodtrx goodtrx
end
%% orgnize .csv trajectory into matlab structure
function [trx] = createTrajectory(flyID,flymatrix,blobColor)

    [flyIDidx] = createFlyIdx(flyID);
    [rowsplit] = createrowsplit(flyIDidx);
    [splitbyID_matrix, splitbyID_blobColor] = createsplitbyID(flyIDidx,flymatrix,rowsplit,blobColor);
    [colorMode]= findcolormode (splitbyID_blobColor,flyIDidx);

    nROW=flyIDidx(end);
    trx=struct([]);


    for i=1:nROW
        trx(i).ID=i-1;
        trx(i).frame=splitbyID_matrix{i}(:,2);
        trx(i).nFlies=splitbyID_matrix{i}(:,3);
        trx(i).flyFrame=splitbyID_matrix{i}(:,4);
        trx(i).blobX=splitbyID_matrix{i}(:,5);
        trx(i).blobY=splitbyID_matrix{i}(:,6);
        trx(i).blobArea=splitbyID_matrix{i}(:,7);
        trx(i).blobAngle=splitbyID_matrix{i}(:,8);
        trx(i).blobA=splitbyID_matrix{i}(:,9);
        trx(i).blobB=splitbyID_matrix{i}(:,10);
        trx(i).blobColor=colorMode{i};
        trx(i).blobdeltaX=splitbyID_matrix{i}(:,11);
        trx(i).blobdeltaY=splitbyID_matrix{i}(:,12);
        orgidx=rowsplit(2*i-1);
        trx(i).originalIdx=flyID(orgidx);

    end
end

function [flyIDidx,flyID] = createFlyIdx(flyID)
    %create numeric index of fly by flyID strings

    nROW=size(flyID,1);
    flyIDidx=zeros(nROW,1);
    flyIDidx(1,1)=1;

    for i=1:nROW-1
        if strcmp(flyID(i),flyID(i+1))==1
            flyIDidx(i+1,1) = flyIDidx(i,1);
        else
            flyIDidx(i+1,1)=flyIDidx(i,1)+1;
        end

    end
end

function [rowsplit] = createrowsplit(flyIDidx)
    %create rowsplit index for each fly accoring to numeric index of fly

    nROW=size(flyIDidx,1);
    rowsplit(1)=1;
        for i=1:nROW-1

            if flyIDidx(i)==flyIDidx(i+1)
                if i==nROW-1
                   rowsplit=[rowsplit;i+1];
                end
                continue;

            else

                rowsplit=[rowsplit;i;i+1];

                if i==nROW-1
                rowsplit=[rowsplit;i+1];    
                end    

            end
        end
end


function [splitbyID_matrix, splitbyID_blobColor] = createsplitbyID(flyIDidx,flymatrix,rowsplit,blobColor)
    %split flymatrix and blobColor by rowsplit index. Each fly are stored in one filed of the
    %cell.
    %sort data by frame

    IDmatrix=[flyIDidx,flymatrix];
    nROW=flyIDidx(end);
    splitbyID_matrix=cell(1,flyIDidx(end));
    splitbyIDunsort=cell(1,flyIDidx(end));
    splitbyID_blobColor=cell(1,flyIDidx(end));

        for i= 1:nROW
        splitbyIDunsort{1,i}= IDmatrix(rowsplit(2*i-1):rowsplit(2*i),:);
        splitbyID_matrix{1,i}=sortrows(splitbyIDunsort{1,i},2); %sort by frame
        splitbyID_blobColor{i}= blobColor(rowsplit(2*i-1):rowsplit(2*i),:);
        end
end

function [colorMode]= findcolormode (splitbyID_blobColor,flyIDidx)

    %this function is used to find out the mode of color of each fly and lable
    %the fly with that color.
    colorMode= cell(1,flyIDidx(end));
    nfly=flyIDidx(end);
    for i=1:nfly
    uniqueColor = unique(splitbyID_blobColor{i});
    n=zeros(length(uniqueColor),1);
        for j= 1: length(uniqueColor)
            n(j)= length(find(strcmp(uniqueColor{j},splitbyID_blobColor{i})));
        end
        [~,jtemp]=max(n); %% still cannot handle ties!!
        colorMode{i}=uniqueColor(jtemp);
    end
end



%% select good trx
function [ goodtrx ] = selectgoodtrx(trx)
    %delete flies <10 frame
    %set flies with nFlies>1 to missing--no don't do this'

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
   % iteri2=size(goodtrx,2);

    %for i=1:iteri2

     %   iterj=size(goodtrx(i).frame);

      %      for j=1:iterj
       %         if  goodtrx(i).nFlies(j)>1
        %            goodtrx(i).blobX(j)=NaN;
         %           goodtrx(i).blobY(j)=NaN;
          %          goodtrx(i).blobArea(j)=NaN;
           %         goodtrx(i).blobAngle(j)=NaN;
            %        goodtrx(i).blobA(j)=NaN;
             %       goodtrx(i).blobB(j)=NaN;
              %      goodtrx(i).blobdeltaX(j)=NaN;
               %     goodtrx(i).blobdeltaY(j)=NaN;
%
 %               end
   %         end




  %  end
end

%%convert blob angle to JAABA theta

function [thetaPtoJpi] = convertThetaPtoJ(goodtrx)
    %this function is used to transform our theata to jabba theta based on
    %two criteria: delta X,deltaY, and one fly cannot swith orientation by 180
    %degree
    % thetaPtoJpi is a m by n cell array with m for single fly n for a single frame  (fly by frame matrix)


    DtoP=pi/180;
    thetaPtoJ={};
    thetaPtoJpi={};
    nfly=size(goodtrx,2);

    for j=1:nfly

            COL=size(goodtrx(j).blobAngle);

            for i=1:COL-1
                if (90<=goodtrx(j).blobAngle(i)) && (goodtrx(j).blobAngle(i)<=180)

                     if goodtrx(j).blobdeltaY(i)<0
                         thetaPtoJ{j}(i)=goodtrx(j).blobAngle(i);
                     elseif goodtrx(j).blobdeltaY(i)>0
                         thetaPtoJ{j}(i)=goodtrx(j).blobAngle(i)-180;
                     else
                         thetaPtoJ{j}(i)=NaN;
                     end

                else %(180<goodtrx(j).blobdeltaY(i)) && (goodtrx(j).blobdeltaY(i)<270);
                     if goodtrx(j).blobdeltaY(i)<0
                         thetaPtoJ{j}(i)=goodtrx(j).blobAngle(i)-180;
                     elseif  goodtrx(j).blobdeltaY(i)>0
                        thetaPtoJ{j}(i)=goodtrx(j).blobAngle(i)-360;
                     else
                        thetaPtoJ{j}(i)=NaN;
                     end

                end
            end

            thetaPtoJ{j}=[thetaPtoJ{j},NaN];

           % for i=1:COL-1
           %     if  abs(thetaPtoJ{j}(i+1)-thetaPtoJ{j}(i))>=90
           %         thetaPtoJ{j}(i+1)=thetaPtoJ{j}(i);
           %     end
           % end

           diffPtoJ{j}=diff(thetaPtoJ{j});
           [COLdiff]=find(abs(diffPtoJ{j})>=170);

           for i= 1:length(COLdiff)

               thetaPtoJ{j}(COLdiff(i))= thetaPtoJ{j}(COLdiff(i))+180;
           end   

            thetaPtoJpi{j} = thetaPtoJ{j} .* DtoP;

    end
end
%% create JAABA input format
function [gpJAABAformat] = create_gpJAABAformat(goodtrx,thetaPtoJpi,x,y,d)
    %UNTITLED4 Summary of this function goes here
    %   Detailed explanation goes here

    gpJAABAformat=([]);
    nFlies=size(goodtrx,2);

    arenaloc=struct('x',x,'y',y,'r',d/2); %  arena center x=395,y=300.5,r=556/2;

    pxpermm= d/40; %556 pixels per 40mm
    meanFrame_per_second=30;


    for i= 1:nFlies

        gpJAABAformat(i).arena=arenaloc;

        gpJAABAformat(i).x= goodtrx(i).blobX'; 
        gpJAABAformat(i).y= goodtrx(i).blobY';


        %x_mm and y_mm are centered by arena center. 
        gpJAABAformat(i).x_mm= (goodtrx(i).blobX'-arenaloc.x) ./ pxpermm; 
        gpJAABAformat(i).y_mm= -1*(goodtrx(i).blobY'-arenaloc.y) ./ pxpermm;


        gpJAABAformat(i).theta=thetaPtoJpi{i};
        gpJAABAformat(i).theta_mm=gpJAABAformat(i).theta;


        gpJAABAformat(i).a=goodtrx(i).blobA' ./4;
        gpJAABAformat(i).b=goodtrx(i).blobB' ./4;

        gpJAABAformat(i).a_mm=gpJAABAformat(i).a ./ pxpermm;
        gpJAABAformat(i).b_mm=gpJAABAformat(i).b ./ pxpermm;

         gpJAABAformat(i).nframes=size(goodtrx(i).flyFrame,1);
         gpJAABAformat(i).firstframe=goodtrx(i).frame(1);
         gpJAABAformat(i).endframe=gpJAABAformat(i).nframes+gpJAABAformat(i).firstframe-1;  % we cannot use goodtrx(i).frame(end) because sometimes there're missing frames there'

                                
         gpJAABAformat(i).off= 1-goodtrx(i).frame(1);

         gpJAABAformat(i).id= goodtrx(i).ID;

         gpJAABAformat(i).fps= meanFrame_per_second;
         gpJAABAformat(i).timestamps= goodtrx(i).frame' ./30;
         gpJAABAformat(i).dt= diff(gpJAABAformat(i).timestamps);

         gpJAABAformat(i).sex='M';%% show blobColor under sex

    %not necessary     gpJAABAformat(i).annname=[];
    %not necessary     gpJAABAformat(i).moviename=movienameinput;
    end
end



