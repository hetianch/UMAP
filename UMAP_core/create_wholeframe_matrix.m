function [ wholeframe_matrix,split_idx ] = create_wholeframe_matrix(wholeframe_cellarray)
%This function is used to convert whole frame window features from JAABA
%into a single matrix (wholeframe_matrix) and also output the split_idx(a vector of the first and end rows of each fly)

wholeframe_matrix=[];
split_idx=[];
for i = 1: size(wholeframe_cellarray,2)
split_idx= [split_idx;size(wholeframe_matrix,1)+1];
attach= wholeframe_cellarray{i};
wholeframe_matrix=[wholeframe_matrix;attach];
split_idx= [split_idx;size(wholeframe_matrix,1)];
end

