function [goodtrx, trx] = create_JAABA_trajectory(flyID,flymatrix,blobColor,x,y,d,frames_second,pixels_mm)
    %This function is used to create JAABA input file : registered_trx.mat from
    %original .csv trajectory file
    [roughtrx] = createTrajectory(flyID,flymatrix,blobColor);
    timestamps=(1:1:max(flymatrix(:,1))) ./frames_second;
    clear flyID flymatrix blobColor;
    [goodtrx] = selectgoodtrx(roughtrx);
    clear roughtrx;
    %[goodtrx_delta] = compute_delta(goodtrx);
    %clear goodtrx;
    [thetaPtoJpi] = convertThetaPtoJ(goodtrx);
    [trx] = create_gpJAABAformat(goodtrx,thetaPtoJpi,x,y,d,frames_second,pixels_mm);
    clear thetaPtoJpi;
    

    save registered_trx timestamps trx;
    save goodtrx goodtrx
end
%% orgnize .csv trajectory into matlab structure
function [trx] = createTrajectory(flyID,flymatrix,blobColor)
    [flyIDidx] = createFlyIdx(flyID);
    [rowsplit] = createrowsplit(flyIDidx);
    [splitbyID_matrix, splitbyID_blobColor] = createsplitbyID(flyIDidx,flymatrix,rowsplit,blobColor);
   % [splitbyID_matrix] = createsplitbyID(flyIDidx,flymatrix,rowsplit);
   % [colorMode]= findcolormode (splitbyID_blobColor,flyIDidx);

    nROW=flyIDidx(end);
    trx=struct([]);


    for i=1:nROW
        trx(i).ID=i-1;
        trx(i).frame=splitbyID_matrix{i}(:,2);
        trx(i).nFlies=splitbyID_matrix{i}(:,3);
       % trx(i).flyFrame=splitbyID_matrix{i}(:,4);
        trx(i).blobX=splitbyID_matrix{i}(:,4);
        trx(i).blobY=splitbyID_matrix{i}(:,5);
        trx(i).blobArea=splitbyID_matrix{i}(:,6);
        trx(i).blobAngle=splitbyID_matrix{i}(:,7);
        trx(i).blobA=splitbyID_matrix{i}(:,8);
        trx(i).blobB=splitbyID_matrix{i}(:,9);
        %trx(i).blobColor=colorMode{i};
        %trx(i).blobdeltaX=splitbyID_matrix{i}(:,11);
        trx(i).blobdeltaY=splitbyID_matrix{i}(:,10);
        orgidx=rowsplit(2*i-1);
        trx(i).originalIdx=flyID(orgidx);
        trx(i).blobColor=splitbyID_blobColor{i};

    end
end

function [flyIDidx] = createFlyIdx(flyID)
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
%function [splitbyID_matrix] = createsplitbyID(flyIDidx,flymatrix,rowsplit)
    %split flymatrix and blobColor by rowsplit index. Each fly are stored in one filed of the
    %cell.
    %sort data by frame
    IDmatrix=[flyIDidx,flymatrix];
    nROW=flyIDidx(end);
    splitbyID_matrix=cell(1,flyIDidx(end));
    splitbyIDunsort=cell(1,flyIDidx(end));
    %splitbyID_blobColor=cell(1,flyIDidx(end));

        for i= 1:nROW
        splitbyIDunsort{1,i}= IDmatrix(rowsplit(2*i-1):rowsplit(2*i),:);
        splitbyID_matrix{1,i}=sortrows(splitbyIDunsort{1,i},2); %sort by frame
        splitbyID_blobColor{i}= blobColor(rowsplit(2*i-1):rowsplit(2*i),:);
        end
end
% % turn this function off temperorily because we can not assume the gender
% doesn't change over trajectory. A more sophisticated way is to find color
% mode within two 'merge'
% function [colorMode]= findcolormode (splitbyID_blobColor,flyIDidx)
% 
%     %this function is used to find out the mode of color of each fly and lable
%     %the fly with that color.
%     colorMode= cell(1,flyIDidx(end));
%     nfly=flyIDidx(end);
%     for i=1:nfly
%     uniqueColor = unique(splitbyID_blobColor{i});
%     n=zeros(length(uniqueColor),1);
%         for j= 1: length(uniqueColor)
%             n(j)= length(find(strcmp(uniqueColor{j},splitbyID_blobColor{i})));
%         end
%         [~,jtemp]=max(n); %% still cannot handle ties!!
%         colorMode{i}=uniqueColor(jtemp);
%     end
% end



%% select good trx
function [ goodtrx ] = selectgoodtrx(trx)
    %delete flies <50 frame
    %set flies with nFlies>1 to missing--no don't do this'

    goodtrx=trx;
    iteri=size(trx,2);
    deletelist=[];


    for i=1:iteri
        %delete flies < 50 frame
        if size(trx(i).frame,1)<50
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

% %%convert blob angle to JAABA theta
% function [goodtrx_delta] = compute_delta(goodtrx)
% %This function is used to compute deltaX an deltaY;
% goodtrx_delta=goodtrx;
% for i=1: size(goodtrx,2);
%     %set all deltaX to nan 
%     dim=length(goodtrx(i).blobX);
%     deltaX=nan(1,dim);
%     deltaY=nan(1,dim);
%     %set second frame of delta
%     if goodtrx(i).nFlies(1)==1 && goodtrx(i).nFlies(2)==1 && abs(goodtrx(i).blobArea(2)-goodtrx(i).blobArea(1))<100
%        deltaX(2)=goodtrx(i).blobX(2)-goodtrx(i).blobX(1);
%        deltaY(2)=goodtrx(i).blobY(2)-goodtrx(i).blobY(1);
%     end
%     
%     for j = 2:dim
%         %only update delta value when certain condition is satisfied!
%         if goodtrx(i).nFlies(j)==1 && ~isnan(deltaX(j-1))
%            if abs(goodtrx(i).blobArea(j)-goodtrx(i).blobArea(j-1))<100 % or framesD>5
%             
%               deltaX(j)=deltaX(j-1)*0.7+(goodtrx(i).blobX(j)-goodtrx(i).blobX(j-1))*0.3;
%            else
%               deltaX(j)=deltaX(j-1);
%            end
%         end
%         
%     end
%     
%      for j = 2:dim
%         %only update delta value when certain condition is satisfied!
%         if goodtrx(i).nFlies(j)==1 && ~isnan(deltaY(j-1))
%            if abs(goodtrx(i).blobArea(j)-goodtrx(i).blobArea(j-1))<100 % or framesD>5
%             
%               deltaY(j)=deltaY(j-1)*0.7+(goodtrx(i).blobY(j)-goodtrx(i).blobY(j-1))*0.3;
%            else
%               deltaY(j)=deltaY(j-1);
%            end
%         end
%         
%      end
%     goodtrx_delta(i).blobdeltaX=deltaX';
%     goodtrx_delta(i).blobdeltaY=deltaY';
%     
% end
% end

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
                         thetaPtoJ{j}(i)=goodtrx(j).blobAngle(i) * (-1);
                     elseif goodtrx(j).blobdeltaY(i)>0
                         thetaPtoJ{j}(i)=180-goodtrx(j).blobAngle(i);
                     else
                         thetaPtoJ{j}(i)=NaN;
                     end

                else %(180<goodtrx(j).blobAngle(i)) && (goodtrx(j).blobAngle(i)<270);
                     if goodtrx(j).blobdeltaY(i)<0
                         thetaPtoJ{j}(i)=180-goodtrx(j).blobAngle(i);
                     elseif  goodtrx(j).blobdeltaY(i)>0
                        thetaPtoJ{j}(i)=360-goodtrx(j).blobAngle(i);
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

%            diffPtoJ{j}=[Nan,diff(thetaPtoJ{j})];
%            [COLdiff]=find(abs(diffPtoJ{j})>=170);
%            
%            for i= 1:length(COLdiff)
% 
%                thetaPtoJ{j}(COLdiff(i))= thetaPtoJ{j}(COLdiff(i))+180;
%            end   

            thetaPtoJpi{j} = thetaPtoJ{j} .* DtoP;

    end
end
%% create JAABA input format
function [gpJAABAformat] = create_gpJAABAformat(goodtrx,thetaPtoJpi,x,y,d,frames_second,pixels_mm)
    %UNTITLED4 Summary of this function goes here
    %   Detailed explanation goes here

    gpJAABAformat=([]);
    nFlies=size(goodtrx,2);

    arenaloc=struct('x',x,'y',y,'r',d/2); %  arena center x=395,y=300.5,r=556/2;

    pxpermm= pixels_mm; %556 pixels per 40mm
    meanFrame_per_second=frames_second;


    for i= 1:nFlies

        gpJAABAformat(i).arena=arenaloc;

        gpJAABAformat(i).x= goodtrx(i).blobX'; 
        gpJAABAformat(i).y= goodtrx(i).blobY';


        %x_mm and y_mm = x and y if no rotation
        gpJAABAformat(i).x_mm= (goodtrx(i).blobX') ./ pxpermm; 
        gpJAABAformat(i).y_mm= (goodtrx(i).blobY') ./ pxpermm;


        gpJAABAformat(i).theta=thetaPtoJpi{i};
        gpJAABAformat(i).theta_mm=gpJAABAformat(i).theta;


        gpJAABAformat(i).a=goodtrx(i).blobA' ./2;
        gpJAABAformat(i).b=goodtrx(i).blobB' ./2;

        gpJAABAformat(i).a_mm=gpJAABAformat(i).a ./ pxpermm;
        gpJAABAformat(i).b_mm=gpJAABAformat(i).b ./ pxpermm;

         gpJAABAformat(i).nframes=size(goodtrx(i).frame,1);
         gpJAABAformat(i).firstframe=goodtrx(i).frame(1);
         gpJAABAformat(i).endframe=goodtrx(i).frame(end);  % note: sometimes there're missing frames there'

                                
         gpJAABAformat(i).off= 1-goodtrx(i).frame(1);

         gpJAABAformat(i).goodtrxID= goodtrx(i).ID;
         gpJAABAformat(i).id= i;
         

         gpJAABAformat(i).fps= meanFrame_per_second;
         gpJAABAformat(i).timestamps= goodtrx(i).frame' ./frames_second;
         gpJAABAformat(i).dt= diff(gpJAABAformat(i).timestamps);

         gpJAABAformat(i).sex='M';%% we set all sex to 'M' since we don't have sex info
         gpJAABAformat(i).color=goodtrx(i).blobColor;

    %not necessary     gpJAABAformat(i).annname=[];
    %not necessary     gpJAABAformat(i).moviename=movienameinput;
    end
end



