function all_window= compute_all_window(trx,perframe,params)
%This function is used to compute window features for all 

all_window=[];
nflies=size(trx,2);

for i= 1:nflies
    all_window(i).fly=i;
    perframe_vector=perframe{i}';
    all_window(i).window=wrap_compute_window(params,perframe_vector);
end

end

