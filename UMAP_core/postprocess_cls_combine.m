function out = postprocess_cls_combine( trx,multiScore, femaleScore,chaseScore,closestfly_anglerange, abs_closestfly_speed,female_str, male_str )
%This function is used to post process combination of multifly sex chase
%classifier's results and genearte behavior structure

[scoredTrx, scores] = attachScores(trx, multiScore,'multiFly');
[multi_female_trx, scores] = attachScores(scoredTrx, femaleScore,'female');
trx_with_sex = impute_gender_with_multiScore(multi_female_trx,female_str,male_str);
save trx_with_sex
[trx_with_sex_chase, scores] = attachScores(trx_with_sex, chaseScore,'chase');
[ behavior] = create_behavior_structure_from_trx( trx_with_sex_chase,closestfly_anglerange,abs_closestfly_speed);
save behavior
[ behavior_final ] = find_chased_gender( behavior,trx );
chase_percent(behavior_final);
save behavior_final




end

