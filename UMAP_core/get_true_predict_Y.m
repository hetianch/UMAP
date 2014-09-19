function [ tru_predict_Y ] = get_true_predict_Y( JAABAscores,gtLabels )
%This function read in JAABAscores (python scores or brad's prediction) and gtLabels, return a
%n by 2 matrix with true  and predict y for each colomns

% x.gtLabels.imp_t0s, t1s, and names all start with an empty cell. So let's
% get rid of empty cells first.
names=gtLabels.names;
t0s=gtLabels.imp_t0s;
t1s=gtLabels.imp_t1s;
flies=gtLabels.flies;

empty_names = cellfun(@isempty,names);
empty_t0s=cellfun(@isempty,t0s);
empty_t1s=cellfun(@isempty,t1s);

names(empty_names) = [];
t0s(empty_t0s)=[];
t1s(empty_t1s)=[];

%%somehow flies always start with 1 even if fly i is not labeled. So we
%%read flies from flies(2) and when we label in ground truth, we never
%%label fly 1.
flies=flies(2:end);

cm_matrix=[];
fly_idx=[];

%I want to construct the following things  first
%fly_idx frame label_true 
%1          33   1  
%1          23   1
%2          23   0

% and then using the first two colomns to generate corresponding:
% label_predict

% So the cm_matrix would look like:

%fly frame label_true label_predict
%1      33   1              1
%1      23   1              0      
%2      23   0              1



for i= 1: length(names)
fly_repeat_idx= size(names{i},2);
fly_idx=[fly_idx;ones(fly_repeat_idx,1)*flies(i)];
end

