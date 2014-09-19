function [ scores_cell_array] = create_python_scores_cellarray(python_scores,split_idx)
%This function is used to convert scores from python package to JAABA
%scores format

% First read in python_scores and split_idx (created by create_wholeframe_matrix.m) to convert scores from python
% to cell array of scores of each fly.

flynumber = size(split_idx,1)/2;
scores_cell_array={};

for i = 1: flynumber
    
    scores_cell_array{i}=python_scores(split_idx(2*i-1):split_idx(2*i))';
end


