function [gender_x,gender_y] = get_gender_x_y(truelabel,speedtrx)
%This function is used to get the features of gender: meanarea, speed,
%speedVariance from speedtrx

emptyCells=cellfun(@isempty,truelabel);
validCells=find(emptyCells==0);
gender_x=[];
gender_y=[];

    for i=validCells

        for j = 1:size(truelabel{i},1)
            gender_x_temp=[];
            gender_y_temp=[];
            
            gender_x_temp(:,1)=speedtrx(i).meanarea((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1))';  
            gender_x_temp(:,2)=speedtrx(i).speed((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1))';
            gender_x_temp(:,3)=speedtrx(i).speedVariance((truelabel{i}(j,1)-speedtrx(i).frame(1)+1):(truelabel{i}(j,2)-speedtrx(i).frame(1)+1))';

            gender_x=[gender_x;gender_x_temp];
            
            gender_y_temp=truelabel{i}(j,3)*ones(truelabel{i}(j,2)-truelabel{i}(j,1)+1,1);
            gender_y=[gender_y;gender_y_temp];
          
        end
    end
end

