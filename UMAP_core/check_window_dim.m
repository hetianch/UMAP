function out  = check_window_dim( window ,trx)
%check window features for all flies created by JAABA : 
flynumber= size(window,2);
for i = 1: flynumber
    nframes=trx(i).nframes;
    window_feature_dim=size(window{i},1);
    if nframes~=window_feature_dim
        fprintf('find dim mismatch for fly %d\n',i);
    end
end

