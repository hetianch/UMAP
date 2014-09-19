
%%These couple of functions are used to create trajectory of flies. These
%%trajectory could be easily transformed to JAABA format.

%%% Step by step guide starts here:
% 1load lab data .csv in by matlab import
% 2sort by flyID %%% this is extremely important!!!!
% 3save blobColor, flyID, blobindex(didn't used that for now), and other
% numeric features in a matrix called flymatrix.
% the colonms in flymatrix are:
%colonms:
%1frame
%2nFlies
%3flyFrame
%4blobX
%5blobY
%6blobArea
%7blobAngle
%8blobA
%9blobB
%10deltaX
%11deltaY
%12flySpeed
%13flyArea
%14flyAspect
%15areaDeviance
%16framesDeviance

%4 create a rough trajectory through createTrajectory
%5 select good trajectories through selectgoodtrx
%6 transform our theta to JAABA theta throughconvertThetaPtoJ
%7 create JAABA trajectory through create_gpJAABAformat

%4/5/6/7 were wrapped in function createRegistered_trx



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

function [rowsplit] = createrowsplit(flyIDidx)
%create rowsplit index for each fly accoring to umeric index of fly

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


function [colorMode]= findcolormode (splitbyID_blobColor,flyIDidx)

%this function is used to find out the modeo of color of each fly and lable
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






