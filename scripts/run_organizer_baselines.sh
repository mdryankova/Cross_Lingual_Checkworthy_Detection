#echo Results for the English Dataset
#python -m baselines.subtask_1a \
#-l english \
#-t data/subtask-1a--english/dataset_train_v1_english.tsv \
#-d data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--target_dir trained_models

#echo Results for the Bulgarian Dataset
#python -m baselines.subtask_1a \
#-l bulgarian \
#-t data/subtask-1a--bulgarian/dataset_train_v1_bulgarian.tsv \
#-d data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--target_dir trained_models

#echo Results for the Turkish Dataset
#python -m baselines.subtask_1a \
#-l turkish \
#-t data/subtask-1a--turkish/normalized/dataset_train_v1_turkish.tsv \
#-d data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--target_dir trained_models

echo Results for the Spanish Dataset
python -m baselines.subtask_1a \
-l spanish \
-t data/subtask-1a--spanish/normalized/dataset_train.tsv \
-d data/subtask-1a--spanish/normalized/dataset_dev.tsv \
--target_dir trained_models

