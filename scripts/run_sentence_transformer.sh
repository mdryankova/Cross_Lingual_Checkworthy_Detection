#python -m experiment \
#--train normal \
#--train_batch_size 8 \
#--test_batch_size 1 \
#--output_dir trained_models/sentence_transformer_all \
#--tokenizer sentence-transformers/quora-distilbert-multilingual \
#--pretrained_model sentence-transformers/quora-distilbert-multilingual \
#--lang english turkish arabic bulgarian spanish \
#--model sentence_transformer \
#--cuda \
#--trainer normal \
#--lr 2e-5 \
#--epochs 3 \
#--adam_epsilon 1e-8 \
#--num_labels 2 \
#--max_seq_len 128 \
#--dropout 0.1 \
#--mode valid
#
#python -m experiment \
#--train normal \
#--train_batch_size 8 \
#--test_batch_size 1 \
#--output_dir trained_models/sentence_transformer_adversarial_all \
#--tokenizer sentence-transformers/quora-distilbert-multilingual \
#--pretrained_model sentence-transformers/quora-distilbert-multilingual \
#--lang english turkish arabic bulgarian spanish \
#--model adversarial_sentence_transformer \
#--cuda \
#--trainer adversarial \
#--lr 2e-5 \
#--epochs 3 \
#--adam_epsilon 1e-8 \
#--num_labels 2 \
#--max_seq_len 128 \
#--dropout 0.1 \
#--mode valid \
#--lambd 1.0 \
#--adversarial
#
#python -m experiment \
#--train normal \
#--train_batch_size 8 \
#--test_batch_size 1 \
#--output_dir trained_models/sentence_transformer_adversarial_all_joint \
#--tokenizer sentence-transformers/quora-distilbert-multilingual \
#--pretrained_model sentence-transformers/quora-distilbert-multilingual \
#--lang english turkish arabic bulgarian spanish \
#--model adversarial_sentence_transformer \
#--cuda \
#--trainer adversarial \
#--lr 2e-5 \
#--epochs 3 \
#--adam_epsilon 1e-8 \
#--num_labels 2 \
#--max_seq_len 128 \
#--dropout 0.1 \
#--mode valid \
#--lambd 1.0

python -m experiment \
--train cv \
--train_batch_size 8 \
--test_batch_size 1 \
--output_dir trained_models/sentence_transformer_cv_all \
--tokenizer sentence-transformers/quora-distilbert-multilingual \
--pretrained_model sentence-transformers/quora-distilbert-multilingual \
--lang english turkish arabic bulgarian spanish \
--model sentence_transformer \
--cuda \
--trainer normal \
--lr 2e-5 \
--epochs 3 \
--adam_epsilon 1e-8 \
--num_labels 2 \
--max_seq_len 128 \
--dropout 0.1 \
--mode valid

python -m experiment \
--train cv \
--train_batch_size 8 \
--test_batch_size 1 \
--output_dir trained_models/sentence_transformer_adversarial_cv_all \
--tokenizer sentence-transformers/quora-distilbert-multilingual \
--pretrained_model sentence-transformers/quora-distilbert-multilingual \
--lang english turkish arabic bulgarian spanish \
--model adversarial_sentence_transformer \
--cuda \
--trainer adversarial \
--lr 2e-5 \
--epochs 3 \
--adam_epsilon 1e-8 \
--num_labels 2 \
--max_seq_len 128 \
--dropout 0.1 \
--mode valid \
--lambd 1.0 \
--adversarial

python -m experiment \
--train cv \
--train_batch_size 8 \
--test_batch_size 1 \
--output_dir trained_models/sentence_transformer_adversarial_all_cv_joint \
--tokenizer sentence-transformers/quora-distilbert-multilingual \
--pretrained_model sentence-transformers/quora-distilbert-multilingual \
--lang english turkish arabic bulgarian spanish \
--model adversarial_sentence_transformer \
--cuda \
--trainer adversarial \
--lr 2e-5 \
--epochs 3 \
--adam_epsilon 1e-8 \
--num_labels 2 \
--max_seq_len 128 \
--dropout 0.1 \
--mode valid \
--lambd 1.0