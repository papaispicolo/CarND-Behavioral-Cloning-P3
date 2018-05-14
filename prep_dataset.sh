rm ./data/project_dataset_log.csv 
## Use data from multiple simulations
#cat ./data/sample_training_data/driving_log_curpath.csv >>  ./data/project_dataset_log.csv
cat ./data/kart_data2/driving_log1_curpath.csv >> ./data/project_dataset_log.csv
cat ./data/kart_data2/driving_log2_curpath.csv >> ./data/project_dataset_log.csv
## Use data from Edge Case simulations multiple times
cat ./data/kart_data3/driving_log4_curpath.csv >>  ./data/project_dataset_log.csv
cat ./data/kart_data3/driving_log4_curpath.csv >>  ./data/project_dataset_log.csv
cat ./data/kart_data3/driving_log4_curpath.csv >>  ./data/project_dataset_log.csv
cat ./data/kart_data3/driving_log4_curpath.csv >>  ./data/project_dataset_log.csv
## Use data from Edge Case simulations multiple times 
cat ./data/kart_data3/driving_log5_curpath.csv >>  ./data/project_dataset_log.csv
cat ./data/kart_data3/driving_log5_curpath.csv >>  ./data/project_dataset_log.csv
cat ./data/kart_data3/driving_log5_curpath.csv >>  ./data/project_dataset_log.csv
cat ./data/kart_data3/driving_log5_curpath.csv >>  ./data/project_dataset_log.csv

cat ./data/sample_training_data/driving_log_curpath.csv >>  ./data/project_dataset_log.csv
#cat ./data/tr_center_lane_driving/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_center_lane_counter_driving/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_left_side_tocenter/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_right_side_to_center/driving_log.csv >> ./data/project_dataset_log.csv
#cat ./data/tr_recovery_side/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_smoth_curve/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_smoth_curve/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_track2_cetner_lane/driving_log.csv >> ./data/project_dataset_log.csv 
cat ./data/tr_curve_2/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_curve_2/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr_curve_2/driving_log.csv >> ./data/project_dataset_log.csv
cat ./data/tr2_curve_2/driving_log.csv >> ./data/project_dataset_log.csv 
cat ./data/tr2_curve_2/driving_log.csv >> ./data/project_dataset_log.csv 
cat ./data/tr2_curve_2/driving_log.csv >> ./data/project_dataset_log.csv 
cat ./data/tr2_curve_2/driving_log.csv >> ./data/project_dataset_log.csv 

wc -l ./data/project_dataset_log.csv 
