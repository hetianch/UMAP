function [ multiclass_x,multiclass_y,TABU,area,diagnosis_matrix] = get_x_y_TABU( whole,truelabel,goodtrx)
%This function is used to get the windowfeatures outcomes and nFlies from TABU corresponding to truelabel
%diagnosis_matrix is:
% flynumber, frame, flyframe,blobArea

emptyCells=cellfun(@isempty,truelabel);
validCells=find(emptyCells==0);
multiclass_x=[];
multiclass_y=[];
area=[];
TABU=[];
diagnosis_matrix=[];
frame=[];
flyframe=[];

    for i=validCells

        for j = 1:size(truelabel{i},1)
           
            multiclass_x_temp=whole{i}((truelabel{i}(j,1)-goodtrx(i).frame(1)+1):(truelabel{i}(j,2)-goodtrx(i).frame(1)+1),:);  
            multiclass_x=[multiclass_x;multiclass_x_temp];
            multiclass_y_temp=truelabel{i}(j,3)*ones(truelabel{i}(j,2)-truelabel{i}(j,1)+1,1);
            multiclass_y=[multiclass_y;multiclass_y_temp];
            TABU_temp=goodtrx(i).nFlies((truelabel{i}(j,1)-goodtrx(i).frame(1)+1):(truelabel{i}(j,2)-goodtrx(i).frame(1)+1)); 
            TABU=[TABU;TABU_temp];
            area_temp=goodtrx(i).blobArea((truelabel{i}(j,1)-goodtrx(i).frame(1)+1):(truelabel{i}(j,2)-goodtrx(i).frame(1)+1)); 
            area=[area;area_temp];
            
            frame_temp=(truelabel{i}(j,1):1:truelabel{i}(j,2))';
            flyframe_temp=((truelabel{i}(j,1)-goodtrx(i).frame(1)+1):1:(truelabel{i}(j,2)-goodtrx(i).frame(1)+1))';
            fly_temp=ones(length(frame_temp),1)*i;
            
            diagnosis_matrix_temp=[fly_temp,frame_temp,flyframe_temp,area_temp,multiclass_y_temp];   
            diagnosis_matrix=[diagnosis_matrix; diagnosis_matrix_temp];
        end
    end
end

