function [flyID, blobColor ,flymatrix] = load_trajectory(filename)

%This function is used to load_trajectory 
[fid message]=fopen(filename,'rt');

data_load=textscan(fid,'%f%f%f%f%f%f%f%f%f%s%s','Delimiter',',','HeaderLines',1,'treatAsEmpty',{'NA'});
flyID=data_load{1,10};
blobColor=data_load{1,11};
flymatrix=[data_load{1,1},data_load{1,2},data_load{1,3},data_load{1,4},data_load{1,5},data_load{1,6}...
           ,data_load{1,7},data_load{1,8},data_load{1,9}];
clear data_load;
end

