rm ./data/project_dataset_log.csv 
sed '1,1d' ./data/sample_training_data/driving_log.csv >> ./data/project_dataset_log.csv
#cat ./data/tr_center_lane_driving/driving_log.csv >> ./data/project_dataset_log.csv
#cat ./data/tr_center_lane_counter_driving/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
#cat ./data/tr_recovery_side/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_smoth_curve/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_smoth_curve/driving_log.csv >> ./data/project_dataset_log.csv
#cat ./data/tr_track2_cetner_lane/driving_log.csv >> ./data/project_dataset_log.csv 
wc -l ./data/project_dataset_log.csv 
