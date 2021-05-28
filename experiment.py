from argparse import ArgumentParser
from pathlib import Path

import pickle

from reader.transformer_reader import CheckWorthyDetectionTask
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import random
from model import TRANSFORMER_MODELS, TRAINER
from config import DATASET_PATHS, RANDOM_SEED, CV_DATASET_PATHS
import csv


def reproducible_set():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def train(args):
    reproducible_set()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_option = args.trainer

    langs = args.lang

    train_datasets = []
    for lang in langs:
        train_path = DATASET_PATHS[lang]['train']
        train_dataset = CheckWorthyDetectionTask(train_path, lang, tokenizer_name=args.tokenizer,
                                                 max_seq_len=args.max_seq_len, preprocess=args.preprocess)

        train_datasets.append(train_dataset)

    train_datasets = ConcatDataset(train_datasets)

    train_data_loader = DataLoader(dataset=train_datasets, batch_size=args.train_batch_size, shuffle=True,
                                   drop_last=True)

    model_name = args.model
    model = TRANSFORMER_MODELS[model_name](args)
    if args.preprocess:
        model.transformer.resize_token_embeddings(len(train_dataset.tokenizer))

    # save model
    saved_folder = output_dir / f'{args.lang}_{model_name}.pt'

    if not saved_folder.exists():
        trainer = TRAINER[trainer_option](args)
        trained_model = trainer.train(dataloader=train_data_loader, model=model)
        torch.save(trained_model.state_dict(), saved_folder)

        if args.cuda:
            torch.cuda.empty_cache()
        del model

    model = TRANSFORMER_MODELS[model_name](args)
    trainer = TRAINER[trainer_option](args)
    if args.preprocess:
        model.transformer.resize_token_embeddings(len(train_dataset.tokenizer))
    if args.cuda:
        model.to(torch.device('cuda'))

    model.load_state_dict(torch.load(saved_folder))

    for lang in langs:
        valid_path = DATASET_PATHS[lang]['dev']
        valid_dataset = CheckWorthyDetectionTask(valid_path, lang, tokenizer_name=args.tokenizer,
                                                 max_seq_len=args.max_seq_len, preprocess=args.preprocess)

        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.test_batch_size, shuffle=False)
        results = trainer.test(dataloader=valid_data_loader, model=model)

        preds = results['labels']
        targs = results['targets']
        probs = results['probs']

        print(f'CLASSIFICATION REPORT VALIDATION {lang}')
        print(classification_report(y_true=targs, y_pred=preds, digits=4))
        print(f'CONFUSION MATRIX VALIDATION')
        print(confusion_matrix(y_true=targs, y_pred=preds))
        print('Probs')
        print(probs)
        print('Preds')
        print(preds)
        print('Targets')
        print(targs)

        model_name = f'{lang}_{args.mode}_{model_name}'
        results_out = Path(args.output_dir) / f'{model_name}.tsv'
        tweet_ids = results['tweet_ids']
        with open(results_out, 'w', newline='') as csvfile:
            reswriter = csv.writer(csvfile, delimiter='\t')
            for idx, topic in enumerate(results['topics']):
                reswriter.writerow([topic, tweet_ids[idx], probs[idx], model_name])


def cv_train(args):
    reproducible_set()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_option = args.trainer

    langs = args.lang

    if args.train:
        for i in range(5):
            train_datasets = []
            for lang in langs:
                train_path = Path(CV_DATASET_PATHS[lang]['train']) / f'train_{i + 1}.tsv'
                train_dataset = CheckWorthyDetectionTask(train_path, lang, tokenizer_name=args.tokenizer,
                                                         max_seq_len=args.max_seq_len, preprocess=args.preprocess)

                train_datasets.append(train_dataset)

            train_datasets = ConcatDataset(train_datasets)

            train_data_loader = DataLoader(dataset=train_datasets, batch_size=args.train_batch_size, shuffle=True)

            model_name = args.model
            model = TRANSFORMER_MODELS[model_name](args)

            if args.preprocess:
                model.transformer.resize_token_embeddings(len(train_dataset.tokenizer))

            # save model
            saved_folder = output_dir / f'{args.lang}_{model_name}_fold_{i + 1}.pt'

            if not saved_folder.exists():
                trainer = TRAINER[trainer_option](args)
                trained_model = trainer.train(dataloader=train_data_loader, model=model)
                torch.save(trained_model.state_dict(), saved_folder)

                if args.cuda:
                    torch.cuda.empty_cache()
                del model

    for lang in langs:
        valid_path = DATASET_PATHS[lang][args.mode]
        valid_dataset = CheckWorthyDetectionTask(valid_path, lang, tokenizer_name=args.tokenizer,
                                                 max_seq_len=args.max_seq_len, preprocess=args.preprocess)

        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.test_batch_size, shuffle=False)

        preds_cv = []
        probs_cv = []

        for i in range(5):
            model = TRANSFORMER_MODELS[model_name](args)

            if args.preprocess:
                model.transformer.resize_token_embeddings(len(train_dataset.tokenizer))

            trainer = TRAINER[trainer_option](args)
            if args.cuda:
                model.to(torch.device('cuda'))

            saved_folder = output_dir / f'{args.lang}_{model_name}_fold_{i + 1}.pt'
            model.load_state_dict(torch.load(saved_folder))

            results = trainer.test(dataloader=valid_data_loader, model=model)

            preds = results['labels']
            probs = results['probs']

            preds_cv.append(preds)
            probs_cv.append(probs)

            if args.cuda:
                torch.cuda.empty_cache()
            del model

        probs_mean = np.mean(np.asarray(probs_cv), axis=0)
        # preds_freq =  np.amax(np.asarray(preds), axis=0)

        model_name_out = f'{lang}_{args.mode}_{model_name}'
        results_out = Path(args.output_dir) / f'{model_name_out}.tsv'
        tweet_ids = results['tweet_ids']
        with open(results_out, 'w', newline='', encoding='utf-8') as csvfile:
            reswriter = csv.writer(csvfile, delimiter='\t')
            for idx, topic in enumerate(results['topics']):
                reswriter.writerow([topic, tweet_ids[idx], probs_mean[idx], model_name_out])


def extract_feats(args):
    reproducible_set()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_option = args.trainer

    langs = args.lang
    model_name = args.model

    for lang in langs:
        valid_path = DATASET_PATHS[lang][args.mode]
        valid_dataset = CheckWorthyDetectionTask(valid_path, lang, tokenizer_name=args.tokenizer,
                                                 max_seq_len=args.max_seq_len, preprocess=args.preprocess)

        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.test_batch_size, shuffle=False)

        for i in range(5):
            model = TRANSFORMER_MODELS[model_name](args)

            if args.preprocess:
                model.transformer.resize_token_embeddings(len(valid_dataset.tokenizer))

            trainer = TRAINER[trainer_option](args)
            if args.cuda:
                model.to(torch.device('cuda'))

            saved_folder = output_dir / f'{i + 1}.pt'
            model.load_state_dict(torch.load(saved_folder))

            results = trainer.test(dataloader=valid_data_loader, model=model)



            if args.cuda:
                torch.cuda.empty_cache()
            del model

            results_out = Path(args.output_dir) / f'{i}_{lang}_features.pkl'
            with open(results_out, 'wb') as f:
                pickle.dump(np.asarray(results['features']), f)

            labels_out = Path(args.output_dir) / f'{i}_{lang}_labels.pkl'
            with open(labels_out, 'wb') as f:
                pickle.dump(np.asarray(results['labels']), f)


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
    parser.add_argument('--train', type=str)
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
    parser.add_argument('--mode', type=str)
    parser.add_argument('--trainer', type=str)
    parser.add_argument('--lambd', type=str)
    parser.add_argument('--language_weight', type=float)
    parser.add_argument('--claim_weight', type=float)
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    args = parser.parse_args()

    if args.train == 'normal':
        print('Normal Training mode is selected.')
        train(args)

    elif args.train == 'cv':
        print('CV Training mode is selected.')
        cv_train(args)

    elif args.train == 'feature':
        print('Feature Extraction')
        extract_feats(args)

    # if args.explain:
    #     print(f'Explaining mode is selected.')
    #     explain()
