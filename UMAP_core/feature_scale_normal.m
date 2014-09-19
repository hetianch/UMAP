function [ Xrescale ] = feature_scale_normal(Xoriginal)
%This function normalize feature by mean and std
%if std=0 then do not rescale that colomn

%% most improtantly sometime inf value cause unkonwn errors for algorithms, so here we impuate inf as 1000* max (after rescale)

ROW=size(Xoriginal,1);
COL=size(Xoriginal,2);
Xrescale= [];

for i = 1:COL
    meanX = nanmean(Xoriginal(~isinf(Xoriginal(:,i)),i));
    %denominator=max(Xoriginal(~isinf(Xoriginal(:,i)),i))-min(Xoriginal(~isinf(Xoriginal(:,i)),i));
    %denominator=abs(max(abs(Xoriginal(~isinf(Xoriginal(:,i)),i))));
    %denominator=nanmedian(Xoriginal(~isinf(Xoriginal(:,i)),i));
    denominator=nanstd(Xoriginal(~isinf(Xoriginal(:,i)),i));
    if denominator ==0
        Xrescale(:,COL)= Xoriginal(:,COL);
    else
     for j = 1: ROW
         if ~isinf(Xoriginal(j,i))
           temp= (Xoriginal(j,i)-meanX)/denominator;
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



