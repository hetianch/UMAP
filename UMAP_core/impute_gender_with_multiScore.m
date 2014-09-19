function trx_with_sex = impute_gender_with_multiScore(multi_female_scored_trx,female_str,male_str)
%This function is used to combine multi-fly,sex classifier and blobColor to
%impute gender.

% rules: (always assign female_str to female, male_str to male if that
% interval is not trust worthy, but use classifier to impute unknown_str.
% 1.assign sex for single-fly frames.
% 2.for single-fly interval if most frequent Color is ?Y? or ?B? and frequency>80% ?>assign that string to this interval.
% 3.for interval, if most frequent Color is not ?Y? or ?B? or Y/B?s frequency <80% ?>use sex classifier.

%trx_with_sex.female_combine: -1 multifly, 0 male, 1 female;

scoredTrx=multi_female_scored_trx;
trx_with_sex=scoredTrx;
flynumber= size(scoredTrx,2);

for i = 1: flynumber

    %initialize values
    multiIdx=find(scoredTrx(i).multiFly);
    color_impute=scoredTrx(i).color;
    color_impute_value= -1*ones(size(scoredTrx(i).color,1),1);
    female= -1*ones(size(scoredTrx(i).color,1),1);
   
    %handle empty multiIdx
    if isempty(multiIdx)
        single_frame_color=color_impute;
        [unique_strings, ~, string_map]=unique(single_frame_color);
        most_common_color=unique_strings(mode(string_map)); 
        nframes=length(color_impute);  
        if (strcmp(most_common_color,female_str)) || (strcmp(most_common_color,male_str))
            most_common_color_freq=length(find(ismember(single_frame_color,most_common_color)))/length( single_frame_color);
            if most_common_color_freq>0.60
                color_impute=repmat(most_common_color,1,nframes);
            else % use sex classifier
                female=scoredTrx(i).female;
            end
        else % use sex classifier
            female=scoredTrx(i).female;
        end
        
        female_idx=find(ismember(color_impute,female_str));
        male_idx=find(ismember(color_impute,male_str));
        color_impute_value(female_idx)=1;
        color_impute_value(male_idx)=0;
        use_sex_classifier_idx=find(color_impute_value==-1);
        color_impute_value(use_sex_classifier_idx)=female(use_sex_classifier_idx);

        trx_with_sex(i).female_combine=color_impute_value';
        
    continue; 
    end
    
    
    for j = 1: length(multiIdx)-1
     
        if multiIdx(j+1)-multiIdx(j)<2
            continue;
        else
        %calculate color mode

            single_frame_color=color_impute(multiIdx(j)+1:multiIdx(j+1)-1);
            [unique_strings, ~, string_map]=unique(single_frame_color);
            most_common_color=unique_strings(mode(string_map)); 
            nframes=multiIdx(j+1)-1-multiIdx(j);

            if (strcmp(most_common_color,female_str)) || (strcmp(most_common_color,male_str))
                most_common_color_freq=length(find(ismember(single_frame_color,most_common_color)))/length( single_frame_color);
                if most_common_color_freq>0.8
                    color_impute(multiIdx(j)+1:multiIdx(j+1)-1)=repmat(most_common_color,1,nframes);
                else % use sex classifier
                    female(multiIdx(j)+1:multiIdx(j+1)-1)=scoredTrx(i).female(multiIdx(j)+1:multiIdx(j+1)-1);
                end
            else % use sex classifier
                female(multiIdx(j)+1:multiIdx(j+1)-1)=scoredTrx(i).female(multiIdx(j)+1:multiIdx(j+1)-1);
            end     
        end 
    end
    %Then we need to handle that the multiIdx doesn't start from 1 or doesn't end at the end frame:
    if multiIdx(1)> 1
    %Ex: multiFly=(0,0,1,1,0,0)
    %    multiIdx=(3,4)
        single_frame_color=color_impute(1:multiIdx(1)-1);
        [unique_strings, ~, string_map]=unique(single_frame_color);
        most_common_color=unique_strings(mode(string_map)); 
        nframes=multiIdx(1)-1;


        if (strcmp(most_common_color,female_str)) || (strcmp(most_common_color,male_str))
            most_common_color_freq=length(find(ismember(single_frame_color,most_common_color)))/length( single_frame_color);
            if most_common_color_freq>0.8
                color_impute(1:multiIdx(1)-1)=repmat(most_common_color,1,nframes);

            else % use sex classifier
                female(1:multiIdx(1)-1)=scoredTrx(i).female(1:multiIdx(1)-1);
            end
        else % use sex classifier
            female(1:multiIdx(1)-1)=scoredTrx(i).female(1:multiIdx(1)-1);
        end     
    end

    if scoredTrx(i).nframes-multiIdx(end)>0
        single_frame_color=color_impute(multiIdx(end)+1:scoredTrx(i).nframes);
        [unique_strings, ~, string_map]=unique(single_frame_color);
        most_common_color=unique_strings(mode(string_map)); 
        nframes=scoredTrx(i).nframes-multiIdx(end);

        if (strcmp(most_common_color,female_str)) || (strcmp(most_common_color,male_str))
            most_common_color_freq=length(find(ismember(single_frame_color,most_common_color)))/length( single_frame_color);
            if most_common_color_freq>0.8
                color_impute(multiIdx(end)+1:scoredTrx(i).nframes)=repmat(most_common_color,1,nframes);
            else % use sex classifier
                female(multiIdx(end)+1:scoredTrx(i).nframes)=scoredTrx(i).female(multiIdx(end)+1:scoredTrx(i).nframes);
            end
        else % use sex classifier
            female(multiIdx(end)+1:scoredTrx(i).nframes)=scoredTrx(i).female(multiIdx(end)+1:scoredTrx(i).nframes);
        end   

    end

      female_idx=find(ismember(color_impute,female_str));
      male_idx=find(ismember(color_impute,male_str));
      color_impute_value(female_idx)=1;
      color_impute_value(male_idx)=0;
      use_sex_classifier_idx=find(color_impute_value==-1);
      color_impute_value(use_sex_classifier_idx)=female(use_sex_classifier_idx);
   
      trx_with_sex(i).female_combine=color_impute_value';

end

