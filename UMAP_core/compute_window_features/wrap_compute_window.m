function [ window_x ] = wrap_compute_window(params,x)
%This functoin is used to compute window features for given vector x


names= fieldnames(params.window);

if isempty(find(ismember(names,'mean'), 1))
    mean_x=[];
else
    mean_x=computeMeanWindow(params,x);
end

if isempty(find(ismember(names,'min'), 1))
   min_x=[];
else
   min_x=computeMinWindow(params,x);
end

if isempty(find(ismember(names,'max'), 1))
    max_x=[];
else
    max_x=computeMaxWindow(params,x);
end

if isempty(find(ismember(names,'std'), 1))
    std_x=[];
else
    std_x=computeStdWindow(params,x);
end

if isempty(find(ismember(names,'diff_neighbor_mean'), 1))
    diff_max_x=[];
else
    diff_max_x=computeDiffNeighborMaxWindow(params,x);
end

if isempty(find(ismember(names,'diff_neighbor_min'), 1))
    diff_min_x=[];
else
    diff_min_x=computeDiffNeighborMinWindow(params,x);
end

if isempty(find(ismember(names,'diff_neighbor_max'), 1))
    diff_mean_x=[];
else
    diff_mean_x=computeDiffNeighborMeanWindow(params,x);
end

if isempty(find(ismember(names,'zscore_neighbors'), 1))
    zscore_x=[];
else
     zscore_x=computeZscoreNeighborsWindow(params,x);
    
end

if isempty(find(ismember(names,'change'), 1))
   change_x=[];
else
    change_x=computeChangeWindow(params,x);
end
   
window_x=[mean_x,min_x,max_x,std_x,diff_max_x,diff_min_x,diff_mean_x,zscore_x,change_x];
end

