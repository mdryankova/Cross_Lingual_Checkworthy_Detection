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

#echo Results for Adversarial on English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all/english_valid_adversarial_sentence_transformer.tsv
#
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
#echo Results for Joint English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_joint/english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_dev_v1_english.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_cv_joint/english_valid_adversarial_sentence_transformer.tsv


#echo Results for Normal on Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/sentence_transformer_all/turkish_valid_english_valid_sentence_transformer.tsv
#
#echo Results CV for Normal on Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/sentence_transformer_cv_all/turkish_valid_sentence_transformer.tsv
#
#echo Results for Adversarial on Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all/turkish_valid_english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Adversarial CV on Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_cv_all/turkish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_joint/turkish_valid_english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_cv_joint/turkish_valid_adversarial_sentence_transformer.tsv


#echo Results for Adversarial on Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all/spanish_valid_bulgarian_valid_arabic_valid_turkish_valid_english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Adversarial CV on Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_cv_all/spanish_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Normal on Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_all/spanish_valid_bulgarian_valid_arabic_valid_turkish_valid_english_valid_sentence_transformer.tsv
#
#echo Results for CV Normal on Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_cv_all/spanish_valid_sentence_transformer.tsv
#
#echo Results for Joint Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_joint/spanish_valid_bulgarian_valid_arabic_valid_turkish_valid_english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_cv_joint/spanish_valid_adversarial_sentence_transformer.tsv

#
#echo Results for Normal on Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_all/arabic_valid_turkish_valid_english_valid_sentence_transformer.tsv
#
#echo Results for Normal CV on Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_cv_all/arabic_valid_sentence_transformer.tsv
#
#echo Results for Adversarial on Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all/arabic_valid_turkish_valid_english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Adversarial CV on Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_cv_all/arabic_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_joint/arabic_valid_turkish_valid_english_valid_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/normalized/dataset_dev.tsv \
#--pred-file-path trained_models/sentence_transformer_adversarial_all_cv_joint/arabic_valid_adversarial_sentence_transformer.tsv


echo Results for Normal on Bulgarian dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
--pred-file-path trained_models/sentence_transformer_all/bulgarian_valid_arabic_valid_turkish_valid_english_valid_sentence_transformer.tsv

echo Results for CV Normal on Bulgarian dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
--pred-file-path trained_models/sentence_transformer_cv_all/bulgarian_valid_sentence_transformer.tsv

echo Results for Adversarial Bulgarian dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
--pred-file-path trained_models/sentence_transformer_adversarial_all/bulgarian_valid_arabic_valid_turkish_valid_english_valid_adversarial_sentence_transformer.tsv

echo Results for Adversarial CV Bulgarian dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
--pred-file-path trained_models/sentence_transformer_adversarial_cv_all/bulgarian_valid_adversarial_sentence_transformer.tsv

echo Results for Joint Bulgarian dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
--pred-file-path trained_models/sentence_transformer_adversarial_all_joint/bulgarian_valid_arabic_valid_turkish_valid_english_valid_adversarial_sentence_transformer.tsv

echo Results for Joint CV Bulgarian dataset
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv \
--pred-file-path trained_models/sentence_transformer_adversarial_all_cv_joint/bulgarian_valid_adversarial_sentence_transformer.tsv
