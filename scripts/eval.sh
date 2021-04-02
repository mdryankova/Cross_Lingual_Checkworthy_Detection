#echo Results for the Ngram Baseline on English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_ngram_baseline_english_dataset_dev_v1_english.tsv
#
#echo Results for the Random Baseline on English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_baseline_english_dataset_dev_v1_english.tsv

#echo Results for the Ngram Baseline on Bulgarian dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_ngram_baseline_bulgarian_dataset_dev_v1_bulgarian.tsv
#
#echo Results for the Baseline on Bulgarian dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_baseline_bulgarian_dataset_dev_v1_bulgarian.tsv

#echo Results for the Random Baseline on Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_baseline_turkish_dataset_dev_v1_turkish.tsv

#echo Results for the Ngram Baseline on Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_ngram_baseline_turkish_dataset_dev_v1_turkish.tsv

#echo Results for the Random Baseline on Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_baseline_spanish_dataset_dev.tsv
#
#echo Results for the Ngram Baseline on Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_ngram_baseline_spanish_dataset_dev.tsv

echo Results for the Sentence BERT on English dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
--pred-file-path trained_models/sentence_transformer_en_dev.tsv