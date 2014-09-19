function [ Xrescale ] = feature_scale_max(Xoriginal)
%This function rescale features by the absolute max (not including inf)
% if max is inf, we chose the second max value as denominator.
% if the entire colomn is zeros, which means denominator=0 (absolute max=0) then
% do not rescale that colomn

%% most improtantly sometime inf value cause unkonwn errors for algorithms, so here we impuate inf as 1000* max (after rescale)


ROW=size(Xoriginal,1);
COL=size(Xoriginal,2);
Xrescale= [];

for i = 1:COL
    %meanX = nanmean(Xoriginal(~isinf(Xoriginal(:,i)),i));
    %denominator=max(Xoriginal(~isinf(Xoriginal(:,i)),i))-min(Xoriginal(~isinf(Xoriginal(:,i)),i));
    denominator=abs(max(abs(Xoriginal(~isinf(Xoriginal(:,i)),i))));
    %denominator=nanmedian(Xoriginal(~isinf(Xoriginal(:,i)),i));
    if denominator ==0
        Xrescale(:,COL)= zeros(ROW,1);
    else
     for j = 1: ROW
         if ~isinf(Xoriginal(j,i))
           temp= (Xoriginal(j,i))/denominator;
             if sign(temp)~=sign(Xoriginal(j,i))
             Xrescale(j,i)= -1 * temp;
             else
             Xrescale(j,i)=temp;
             end
         end
     end
    end
end

%% handel inf value here

for k=1:COL

max_after_rescale = abs(max(abs(Xrescale(~isinf(Xrescale(:,k)),k))));
Xrescale(isinf(Xoriginal(:,k)),k)=max_after_rescale*1000;

end



