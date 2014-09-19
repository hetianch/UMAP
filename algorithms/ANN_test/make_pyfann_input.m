function out = make_pyfann_input(feature,label,cv_fold,output_cat)
%This function is usded to prepare pyfann input

% this file is writen in tab deliminated .txt file. After that we need to
% rename .txt to .train or .test

%% format of pyfann input file: 

% 4 2 1   (4: 4 obs, 2: 2 column for each obs, 1: one output) 
% -1 -1
% -1
% -1 1
% 1
% 1 -1
% 1
% 1 1
% -1

%% configure cross validation
file_num=cv_fold;
rows=floor(length(label)/cv_fold);
split=(1:rows:length(label));
all_matrix=[feature,label];
% configure lines for fprint
string='';

for i = 1: size(feature,2)-1
    string_temp='%f\t';
    string=strcat(string,string_temp);   
end
string_final=strcat(string,'%f\n');

for i = 1:file_num-1
    cv_matrix_train=[];
    cv_matrix_test=[];
    obs_num_train= split(i+1)-split(i)+1;
    obs_num_test= size(feature,1)-obs_num_train;
    column_per_obs=size(feature,2);
    
    cv_matrix_test=all_matrix(split(i):split(i+1),:);
    cv_matrix_train=all_matrix;
    cv_matrix_train(split(i):split(i+1),:)=[];
    
    file_name_train=strcat('cv_train',num2str(i),'.train');
    file_name_test=strcat('cv_test',num2str(i),'.test');
    
    file_name_matrix_train=strcat('cv_train',num2str(i),'.txt');
    file_name_matrix_test=strcat('cv_test',num2str(i),'.txt');
    
    %output train and test data in matrix first
    dlmwrite(file_name_matrix_train,cv_matrix_train)
    dlmwrite(file_name_matrix_test,cv_matrix_test)
    
    fileID_train=fopen(file_name_train,'w');
    %write the first line
    fprintf(fileID_train,'%s\t %s\t %s\n',num2str(obs_num_train),num2str(column_per_obs),num2str(output_cat));
    %write the rest lines
    for j = 1: size(cv_matrix_train,1)
        fprintf(fileID_train,string_final,cv_matrix_train(j,1:end-1));
        fprintf(fileID_train,'%d\n',cv_matrix_train(j,end));
    end
    fclose(fileID_train);
    
    fileID_test=fopen(file_name_test,'w');
    %write the first line
    fprintf(fileID_test,'%s\t %s\t %s\n',num2str(obs_num_test),num2str(column_per_obs),num2str(output_cat));
    %write the rest lines
    for k = 1: size(cv_matrix_test,1)
        fprintf(fileID_test,string_final,cv_matrix_test(k,1:end-1));
        fprintf(fileID_test,'%d\n',cv_matrix_test(k,end));
    end
    fclose(fileID_test);
end

%handel last cv file
    cv_matrix_train=[];
    cv_matrix_test=[];
    obs_num= size(feature,1)-split(file_num-1)+1;
    obs_num_test= size(feature,1)-obs_num_train;
    column_per_obs=size(feature,2);
 
    cv_matrix_train=all_matrix(split(file_num-1):end,:);
    cv_matrix_test=all_matrix;
    cv_matrix_test(split(file_num-1):end,:)=[];
    
    file_name_train=strcat('cv_train',num2str(file_num),'.train');
    file_name_test=strcat('cv_test',num2str(file_num),'.test');
    file_name_matrix_train=strcat('cv_train',num2str(file_num),'.txt');
    file_name_matrix_test=strcat('cv_test',num2str(file_num),'.txt');
    
    %output train and test data in matrix first
    dlmwrite(file_name_matrix_train,cv_matrix_train)
    dlmwrite(file_name_matrix_test,cv_matrix_test)
    
    
    fileID_train=fopen(file_name_train,'w');
    %write the first line
    fprintf(fileID_train,'%s %s %s\n',num2str(obs_num_train),num2str(column_per_obs),num2str(output_cat));
    for j = 1: size(cv_matrix_train,1)
    %write the rest lines
        fprintf(fileID_train,string_final,cv_matrix_train(j,1:end-1));
        fprintf(fileID_train,'%d\n',cv_matrix_train(j,end));
    end
    fclose(fileID_train);
    
    fileID_test=fopen(file_name_test,'w');
    %write the first line
    fprintf(fileID_test,'%s %s %s\n',num2str(obs_num_test),num2str(column_per_obs),num2str(output_cat));
    %write the rest lines 
    for k = 1: size(cv_matrix_test,1)
        fprintf(fileID_test,string_final,cv_matrix_test(k,1:end-1));
        fprintf(fileID_test,'%d\n',cv_matrix_test(k,end));
    end
    fclose(fileID_test);
end   
    
