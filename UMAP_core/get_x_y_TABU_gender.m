function [ gender_x,female,TABU,area,diagnosis_matrix] = get_x_y_TABU_gender(truelabel,speedtrx)
%This function is used to get the windowfeatures outcomes and nFlies from TABU corresponding to truelabel
%diagnosis_matrix is:
% flynumber, frame, flyframe,blobArea

emptyCells=cellfun(@isempty,truelabel);
validCells=find(emptyCells==0);
gender_x=[];
female=[];
area=[];
TABU=[];
diagnosis_matrix=[];
frame=[];
flyframe=[];

    for i=validCells

        for j = 1:size(truelabel{i},1)
            gender_x_temp=[];
            female_temp=[];
            TABU_temp=[];
            area_temp=[];
            
            gender_x_temp(:,1)=speedtrx(i).blobArea((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1));  
            gender_x_temp(:,2)=speedtrx(i).speed((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1));
            deltaX= speedtrx(i).blobdeltaX((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1));
            deltaY= speedtrx(i).blobdeltaY((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1));
            delta=sqrt(deltaX .^2+deltaY .^2);
           
            gender_x_temp(:,3)= delta;
            gender_x=[gender_x;gender_x_temp];
            
            female_temp=truelabel{i}(j,3)*ones(truelabel{i}(j,2)-truelabel{i}(j,1)+1,1);
            female=[female;female_temp];
            
            TABU_temp=speedtrx(i).nFlies((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1)); 
            TABU=[TABU;TABU_temp];
            
            area_temp=speedtrx(i).blobArea((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1)); 
            area=[area;area_temp];
            
            frame_temp=(truelabel{i}(j,1):1:truelabel{i}(j,2))';
            flyframe_temp=((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):1:(truelabel{i}(j,2)-speedtrx(i).frame(1)+1))';
            fly_temp=ones(length(frame_temp),1)*i;
            
            diagnosis_matrix_temp=[fly_temp,frame_temp,flyframe_temp,area_temp,gender_x_temp(:,2),female_temp];   
            diagnosis_matrix=[diagnosis_matrix; diagnosis_matrix_temp];
        end
    end
end

