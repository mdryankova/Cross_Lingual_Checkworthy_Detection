from argparse import ArgumentParser
from pathlib import Path
from reader.transformer_reader import CheckWorthyDetectionTask
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import random
from model import TRANSFORMER_MODELS, Trainer
from config import DATASET_PATHS, RANDOM_SEED
import csv


def train(args):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang = args.lang[0]
    train_path = DATASET_PATHS[lang]['train']
    train_dataset = CheckWorthyDetectionTask(train_path, lang, tokenizer_name=args.tokenizer,
                                             max_seq_len=args.max_seq_len)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                   drop_last=True)

    model = TRANSFORMER_MODELS['sentence_transformer'](args)
    model.transformer.resize_token_embeddings(len(train_dataset.tokenizer))

    # save model
    saved_folder = output_dir / f'{args.lang}_sentence_transformer.pt'

    print(saved_folder)

    if not saved_folder.exists():
        trainer = Trainer(args)
        trained_model = trainer.train(dataloader=train_data_loader, model=model)
        torch.save(trained_model.state_dict(), saved_folder)

        if args.cuda:
            torch.cuda.empty_cache()
        del model

    valid_path = DATASET_PATHS[lang]['dev']
    valid_dataset = CheckWorthyDetectionTask(valid_path, lang, tokenizer_name=args.tokenizer,
                                             max_seq_len=args.max_seq_len)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = TRANSFORMER_MODELS['sentence_transformer'](args)
    trainer = Trainer(args)
    model.transformer.resize_token_embeddings(len(valid_dataset.tokenizer))

    if args.cuda:
        model.to(torch.device('cuda'))

    model.load_state_dict(torch.load(saved_folder))
    results = trainer.test(dataloader=valid_data_loader, model=model)

    preds = results['labels']
    targs = results['targets']
    probs = results['probs']

    print(f'CLASSIFICATION REPORT VALIDATION')
    print(classification_report(y_true=targs, y_pred=preds, digits=4))
    print(f'CONFUSION MATRIX VALIDATION')
    print(confusion_matrix(y_true=targs, y_pred=preds))
    print('Probs')
    print(probs)
    print('Preds')
    print(preds)
    print('Targets')
    print(targs)

    model_name = f'{args.lang}_sentence_transformer'
    results_out = Path(args.output_dir) / args.clef_filename
    tweet_ids = results['tweet_ids']
    with open(results_out, 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter='\t')
        for idx, topic in enumerate(results['topics']):
            reswriter.writerow([topic, tweet_ids[idx], probs[idx], model_name])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', help='Enter the directory to save the trained models')
    parser.add_argument('--pretrained_model', help='Pretrained Model of Transformers')
    parser.add_argument('--tokenizer', help='Transformer Tokenizer')
    parser.add_argument('--model', help='Enter model')
    parser.add_argument('--task', help='Enter model', choices=['exist', 'pan_hatespeech'])
    parser.add_argument('--output_dir')
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--input_mode', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_seq_len', type=int)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--attention_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--cuda', action='store_true', help='Indicate whether you use cpu or gpu')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--lang', nargs='+', type=str)
    parser.add_argument('--adam_epsilon', type=float)
    parser.add_argument('--clef_filename', type=str)
    args = parser.parse_args()

    if args.train:
        print('Training mode is selected.')
        train(args)

    # if args.explain:
    #     print(f'Explaining mode is selected.')
    #     explain()
