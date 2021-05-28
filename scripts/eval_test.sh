#echo Results for Joint CV Spanish dataset normal
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/normal_16/spanish_test_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.9-0.1 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/0.9-0.1_16/spanish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.8-0.2 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/0.8-0.2_16/spanish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.7-0.3 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/0.7-0.3_16/spanish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.6-0.4 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/0.6-0.4_16/spanish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.5-0.5 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/0.5-0.5_16/spanish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.4-0.6 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/0.4-0.6_16/spanish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Spanish dataset 0.3-0.7 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
#--pred-file-path trained_models/0.3-0.7_16/spanish_test_adversarial_sentence_transformer.tsv

echo Results for Spanish BETO normal
python -m scorer.subtask_1a \
--gold-file-path data/subtask-1a--spanish/dataset_test_goldstandard.tsv \
--pred-file-path trained_models/spanish_beto/spanish_test_sentence_transformer.tsv


#echo Results for Joint CV English dataset normal with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/normal_16/english_test_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.9-0.1 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/0.9-0.1_16/english_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.8-0.2 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/0.8-0.2_16/english_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.7-0.3 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/0.7-0.3_16/english_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.6-0.4 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/0.6-0.4_16/english_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.5-0.5 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/0.5-0.5_16/english_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.4-0.6 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/0.4-0.6_16/english_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV English dataset 0.3-0.7 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/0.3-0.7_16/english_test_adversarial_sentence_transformer.tsv


#echo Results for Joint English single task
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--english/dataset_test_english.tsv \
#--pred-file-path trained_models/english_sbert/english_test_sentence_transformer.tsv


#echo Results for Joint CV Turkish dataset normal with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/normal_16/turkish_test_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.9-0.1 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/0.9-0.1_16/turkish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.8-0.2 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/0.8-0.2_16/turkish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.7-0.3 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/0.7-0.3_16/turkish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.6-0.4 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/0.6-0.4_16/turkish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.5-0.5 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/0.5-0.5_16/turkish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.4-0.6 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/0.4-0.6_16/turkish_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Turkish dataset 0.3-0.7 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/0.3-0.7_16/turkish_test_adversarial_sentence_transformer.tsv

#echo Results for Turkish dataset single
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--turkish/dataset_test_v1_turkish_labels.tsv \
#--pred-file-path trained_models/turkish_bert_cased/turkish_test_sentence_transformer.tsv


#echo Results for Joint CV Bulgarian dataset normal with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/normal_16/bulgarian_test_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.9-0.1 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/0.9-0.1_16/bulgarian_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.8-0.2 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/0.8-0.2_16/bulgarian_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.7-0.3 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/0.7-0.3_16/bulgarian_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.6-0.4 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/0.6-0.4_16/bulgarian_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.5-0.5 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/0.5-0.5_16/bulgarian_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.4-0.6 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/0.4-0.6_16/bulgarian_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Bulgarian dataset 0.3-0.7 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/0.3-0.7_16/bulgarian_test_adversarial_sentence_transformer.tsv

#echo Results for Bulgarian single Task
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--bulgarian/bulgarian_test_data.tsv \
#--pred-file-path trained_models/bulgarian_bert_cased/bulgarian_test_sentence_transformer.tsv



#echo Results for Joint CV Arabic dataset normal with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/normal_16/arabic_test_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.9-0.1 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/0.9-0.1_16/arabic_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.8-0.2 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/0.8-0.2_16/arabic_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.7-0.3 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/0.7-0.3_16/arabic_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.6-0.4 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/0.6-0.4_16/arabic_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.5-0.5 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/0.5-0.5_16/arabic_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.4-0.6 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/0.4-0.6_16/arabic_test_adversarial_sentence_transformer.tsv
#
#echo Results for Joint CV Arabic dataset 0.3-0.7 with batch 16
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/0.3-0.7_16/arabic_test_adversarial_sentence_transformer.tsv
#
#
#echo results Arabic Single Task
#python -m scorer.subtask_1a \
#--gold-file-path data/subtask-1a--arabic/CT21-AR-Test-T1-Labels.tsv \
#--pred-file-path trained_models/arabic_bert/arabic_test_sentence_transformer.tsv

