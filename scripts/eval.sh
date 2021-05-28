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
#
#echo Results for the Ngram Baseline on Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_ngram_baseline_turkish_dataset_dev_v1_turkish.tsv

echo Results for the Random Baseline on Spanish dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
--pred-file-path trained_models/baseline_subtask_1a_baseline_spanish_dataset_dev.tsv

echo Results for the Ngram Baseline on Spanish dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
--pred-file-path trained_models/baseline_subtask_1a_ngram_baseline_spanish_dataset_dev.tsv

#echo Results for the Random Baseline on Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_baseline_arabic_dev.tsv
#
#echo Results for the Ngram Baseline on Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/baseline_subtask_1a_ngram_baseline_arabic_dev.tsv

#echo Results for Adversarial on English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all/english_valid_adversarial_sentence_transformer.tsv

#echo Results for Adversarial CV on English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_cv_all/english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Normal on English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/sentence_transformer_all/english_valid_sentence_transformer.tsv
#
#echo Results for Normal CV on English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/sentence_transformer_cv_all/english_valid_sentence_transformer.tsv
#
#echo Results for Normal CV on English dataset 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/normal_16/english_valid_sentence_transformer.tsv
#
#
#echo Results for Normal CV on English dataset 16 batch with preprocess
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/normal_16_with_preprocess/english_valid_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.4-0.6
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.4-0.6/english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.5-0.5
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.5-0.5/english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.6-0.4
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.6-0.4/english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.6-0.4 with 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.6-0.4_16/dev_results/english_valid_adversarial_sentence_transformer.tsv
#
##echo Results for Joint CV English dataset 0.6-0.4 with 16 with preprocess
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
##--pred-file-path trained_models/0.6-0.4_16_with_preprocess/english_valid_adversarial_sentence_transformer.tsv
#
#
#echo Results for Joint CV English dataset 0.7-0.3 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.7-0.3_16/english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.7-0.3 32 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.7-0.3_32/english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.7-0.3
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.7-0.3/english_valid_adversarial_sentence_transformer.tsv
#
#
#echo Results for Joint CV English dataset 0.3-0.7
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.3-0.7_16/english_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.4-0.6
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.4-0.6_16/english_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.5-0.5
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.5-0.5_16/english_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.8-0.2
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.8-0.2_16/english_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.9-0.1
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/0.9-0.1_16/english_dev_adversarial_sentence_transformer.tsv
##
##
#echo Results for Normal CV Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/sentence_transformer_cv_all/turkish_valid_sentence_transformer.tsv
#
##echo Results for Normal CV Turkish dataset 16 batch with preprocess
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
##--pred-file-path trained_models/normal_16_with_preprocess/turkish_valid_sentence_transformer.tsv
#
#
##echo Results for Normal CV Turkish dataset 16 batch
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
##--pred-file-path trained_models/normal_16/turkish_valid_sentence_transformer.tsv
#

#echo Results for Joint CV Turkish dataset 0.3-0.7
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.3-0.7_16/turkish_dev_adversarial_sentence_transformer.tsv
##
#
#echo Results for Joint CV Turkish dataset 0.4-0.6
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.4-0.6_16/turkish_dev_adversarial_sentence_transformer.tsv
##
##
#echo Results for Joint CV Turkish dataset 0.5-0.5
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.5-0.5_16/turkish_dev_adversarial_sentence_transformer.tsv
#
##echo Results for Joint CV Turkish dataset 0.6-0.4
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
##--pred-file-path trained_models/0.6-0.4/turkish_valid_adversarial_sentence_transformer.tsv
##
#echo Results for Joint CV Turkish dataset 0.6-0.4 batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.6-0.4_16/dev_results/turkish_valid_adversarial_sentence_transformer.tsv
#
#
#echo Results for Joint CV Turkish dataset 0.8-0.2
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.8-0.2_16/turkish_dev_adversarial_sentence_transformer.tsv
#
#
#echo Results for Joint CV Turkish dataset 0.9-0.1
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.9-0.1_16/turkish_dev_adversarial_sentence_transformer.tsv

#
##echo Results for Joint CV Turkish dataset 0.6-0.4 batch 16 with preprocess
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
##--pred-file-path trained_models/0.6-0.4_16_with_preprocess/turkish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.7-0.3 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.7-0.3_16/turkish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.7-0.3 32 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.7-0.3_32/turkish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.7-0.3
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.7-0.3/turkish_valid_adversarial_sentence_transformer.tsv
#
#
#echo Results for Joint CV Turkish dataset 0.8-0.2
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.8-0.2_16/turkish_dev_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Turkish dataset 0.9-0.1
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/0.9-0.1/turkish_valid_adversarial_sentence_transformer.tsv
##
#echo Results for CV Normal on Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_cv_all/spanish_valid_sentence_transformer.tsv
#
#echo Results for CV Normal on Spanish dataset batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/normal_16/spanish_valid_sentence_transformer.tsv

echo Results for Joint CV Spanish dataset 0.3-0.7
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
--pred-file-path trained_models/0.3-0.7_16/spanish_dev_adversarial_sentence_transformer.tsv


echo Results for Joint CV Spanish dataset 0.4-0.6
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
--pred-file-path trained_models/0.4-0.6_16/spanish_dev_adversarial_sentence_transformer.tsv

echo Results for Joint CV Spanish dataset 0.5-0.5
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
--pred-file-path trained_models/0.5-0.5_16/spanish_dev_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Spanish dataset 0.6-0.4
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/0.6-0.4/spanish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.6-0.4 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/0.6-0.4_16/dev_results/spanish_valid_adversarial_sentence_transformer.tsv
#
#
##echo Results for Joint CV Spanish dataset 0.6-0.4 with batch 16 with preprocess
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
##--pred-file-path trained_models/0.6-0.4_16_with_preprocess/spanish_valid_adversarial_sentence_transformer.tsv
##
#echo Results for Joint CV Spanish dataset 0.7-0.3 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/0.7-0.3_16/spanish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.8-0.2 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/0.8-0.2_16/spanish_dev_adversarial_sentence_transformer.tsv


#echo Results for Joint CV Spanish dataset 0.7-0.3
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/0.7-0.3/spanish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.7-0.3 32 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/0.7-0.3_32/spanish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.8-0.2
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/0.8-0.2/spanish_valid_adversarial_sentence_transformer.tsv

echo Results for Joint CV Spanish dataset 0.9-0.1
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
--pred-file-path trained_models/0.9-0.1_16/spanish_dev_adversarial_sentence_transformer.tsv

#
#echo Results for Normal CV on Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/sentence_transformer_cv_all/arabic_valid_sentence_transformer.tsv

#echo Results for Normal CV on Arabic dataset 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/normal_16/arabic_valid_sentence_transformer.tsv
#

#echo Results for Joint CV Arabic dataset 0.3-0.7
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.3-0.7_16/arabic_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.4-0.6
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.4-0.6_16/arabic_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.5-0.5
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.5-0.5_16/arabic_dev_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Arabic dataset 0.9-0.1
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.9-0.1_16/arabic_dev_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Arabic dataset 0.6-0.4
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.6-0.4/arabic_valid_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Arabic dataset 0.6-0.4 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.6-0.4_16/dev_results/arabic_valid_adversarial_sentence_transformer.tsv
#
##echo Results for Joint CV Arabic dataset 0.6-0.4 16 batch with preprocess
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
##--pred-file-path trained_models/0.6-0.4_16_with_preprocess/arabic_valid_adversarial_sentence_transformer.tsv
#
#
#echo Results for Joint CV Arabic dataset 0.7-0.3 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.7-0.3_16/arabic_valid_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Arabic dataset 0.7-0.3 32 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.7-0.3_32/arabic_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.7-0.3
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.7-0.3/arabic_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.8-0.2
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.8-0.2_16/arabic_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.9-0.1
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dev.tsv \
#--pred-file-path trained_models/0.9-0.1/arabic_valid_adversarial_sentence_transformer.tsv
#
#
##echo Results for Normal CV Bulgarian dataset
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
##--pred-file-path trained_models/sentence_transformer_cv_all/bulgarian_valid_sentence_transformer.tsv
#
#echo Results for Normal CV Bulgarian dataset batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/normal_16/bulgarian_valid_sentence_transformer.tsv
#
##echo Results for Joint CV Bulgarian dataset 0.4-0.6
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
##--pred-file-path trained_models/0.4-0.6/bulgarian_valid_adversarial_sentence_transformer.tsv
##
##echo Results for Joint CV Bulgarian dataset 0.5-0.5
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
##--pred-file-path trained_models/0.5-0.5/bulgarian_valid_adversarial_sentence_transformer.tsv
##
##echo Results for Joint CV Bulgarian dataset 0.6-0.4
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
##--pred-file-path trained_models/0.6-0.4/bulgarian_valid_adversarial_sentence_transformer.tsv
#
##echo Results for Joint CV Bulgarian dataset 0.6-0.4 with batch 16 with preprocess
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
##--pred-file-path trained_models/0.6-0.4_16_with_preprocess/bulgarian_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.6-0.4 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.6-0.4_16/dev_results/bulgarian_valid_adversarial_sentence_transformer.tsv
#
##echo Results for Joint CV Bulgarian dataset 0.7-0.3
##python -m scorer.subtask_1a \
##--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
##--pred-file-path trained_models/0.7-0.3/bulgarian_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.7-0.3 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.7-0.3_16/bulgarian_valid_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Bulgarian dataset 0.8-0.2 16 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.8-0.2_16/bulgarian_dev_adversarial_sentence_transformer.tsv

#echo Results for Joint CV Bulgarian dataset 0.7-0.3 32 batch
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.7-0.3_32/bulgarian_valid_adversarial_sentence_transformer.tsv
#

#echo Results for Joint CV Bulgarian dataset 0.3-0.7
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.3-0.7_16/bulgarian_dev_adversarial_sentence_transformer.tsv
##

#echo Results for Joint CV Bulgarian dataset 0.4-0.6
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.4-0.6_16/bulgarian_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.5-0.5
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.5-0.5_16/bulgarian_dev_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.8-0.2
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.8-0.2_16/bulgarian_dev_adversarial_sentence_transformer.tsv
##
#echo Results for Joint CV Bulgarian dataset 0.9-0.1
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
#--pred-file-path trained_models/0.9-0.1_16/bulgarian_dev_adversarial_sentence_transformer.tsv
#

