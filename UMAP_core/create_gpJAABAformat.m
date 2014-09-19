function [gpJAABAformat] = create_gpJAABAformat(goodtrx,thetaPtoJpi)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

gpJAABAformat=([]);
nFlies=size(goodtrx,2);

arenaloc=struct('x',395,'y',300.5,'r',278); %  arena center x=395,y=300.5,r=556/2;

pxpermm= 556/40; %556pixels per 40mm
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
     gpJAABAformat(i).endframe=goodtrx(i).frame(end);
     
     gpJAABAformat(i).off= 1-goodtrx(i).frame(1);
     
     gpJAABAformat(i).id= goodtrx(i).ID;
     
     gpJAABAformat(i).fps= meanFrame_per_second;
     gpJAABAformat(i).timestamps= goodtrx(i).frame' ./30;
     gpJAABAformat(i).dt= diff(gpJAABAformat(i).timestamps);
     
     gpJAABAformat(i).sex=goodtrx.blobColor;%% show blobColor under sex

%not necessary     gpJAABAformat(i).annname=[];
%not necessary     gpJAABAformat(i).moviename=movienameinput;
end

