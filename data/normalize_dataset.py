import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import json


def normalize_turkish_dataset(dpath, tpath):
    data = pd.read_csv(dpath, sep='\t')
    data['claim'] = 'no_claim'
    data.rename(columns={"check-worthiness": "check_worthiness"}, inplace=True)
    data = data[['topic_id', 'tweet_id', 'tweet_url', 'tweet_text', 'claim', 'check_worthiness']]
    data.to_csv(tpath, sep='\t', index=False)


def normalize_spanish_dataset(dpath, tpath):
    data = pd.read_csv(dpath, sep='\t')
    data.rename(columns={"check-worthiness": "check_worthiness"}, inplace=True)
    data = data[['topic_id', 'tweet_id', 'tweet_url', 'tweet_text', 'claim', 'check_worthiness']]
    data.to_csv(tpath, sep='\t', index=False)


def normalize_english_dataset(dpath, tpath):
    data = pd.read_csv(dpath, sep='\t')
    data.rename(columns={"claim_worthiness": "check_worthiness"}, inplace=True)
    data = data[['topic_id', 'tweet_id', 'tweet_url', 'tweet_text', 'claim', 'check_worthiness']]
    data.to_csv(tpath, sep='\t', index=False)


def normalize_arabic_dataset(dpath, tpath):
    output_dir = Path(tpath)
    data = []
    with open(dpath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            splits = line.split('\t')
            data.append({
                'topic_id': splits[0],
                'tweet_id': int(splits[1]),
                'tweet_url': splits[2],
                'tweet_text': splits[3],
                'claim': splits[4],
                'check_worthiness': int(splits[5])
            })

    data = pd.DataFrame(data)
    train, valid = train_test_split(data, test_size=0.20, random_state=42, stratify=data['check_worthiness'].tolist())

    train.to_csv(output_dir / 'dataset_train.tsv', sep='\t', index=False)
    valid.to_csv(output_dir / 'dataset_dev.tsv', sep='\t', index=False)


def create_cv_folds(src_file, target_file):
    data = pd.read_csv(src_file, sep='\t')
    labels = data['check_worthiness'].tolist()
    skf = StratifiedKFold(n_splits=5)
    target_dir = Path(target_file)

    for idx, (train_index, test_index) in enumerate(skf.split(data, labels)):
        X_train, X_test = data.loc[train_index], data.loc[test_index]
        X_train.to_csv(target_dir / f'train_{idx + 1}.tsv', sep='\t', index=False)
        X_test.to_csv(target_dir / f'test_{idx + 1}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    # normalize_turkish_dataset('./subtask-1a--turkish/dataset_train_v1_turkish.tsv',
    #                           './subtask-1a--turkish/normalized/dataset_train_v1_turkish.tsv')
    # normalize_turkish_dataset('./subtask-1a--turkish/dataset_dev_v1_turkish.tsv',
    #                           './subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv')
    #
    # normalize_spanish_dataset('./subtask-1a--spanish/dataset_train.tsv',
    #                           './subtask-1a--spanish/normalized/dataset_train.tsv')
    # normalize_spanish_dataset('./subtask-1a--spanish/dataset_dev.tsv',
    #                           './subtask-1a--spanish/normalized/dataset_dev.tsv')

    # normalize_english_dataset('./subtask-1a--english/dataset_train_v1_english.tsv',
    #                           './subtask-1a--english/normalized/dataset_train.tsv')
    # normalize_english_dataset('./subtask-1a--english/dataset_dev_v1_english.tsv',
    #                           './subtask-1a--english/normalized/dataset_dev.tsv')
    #
    # normalize_arabic_dataset('./subtask-1a--arabic/CT21-AR-Train-T1-Labels.txt',
    #                          './subtask-1a--arabic/normalized/')

    create_cv_folds('./subtask-1a--turkish/normalized/dataset_train_v1_turkish.tsv', './subtask-1a--turkish/folds/')
    create_cv_folds('./subtask-1a--arabic/normalized/dataset_train.tsv', './subtask-1a--arabic/folds/')
    create_cv_folds('./subtask-1a--spanish/normalized/dataset_train.tsv', './subtask-1a--spanish/folds/')
    create_cv_folds('./subtask-1a--english/normalized/dataset_train.tsv', './subtask-1a--english/folds/')
    create_cv_folds('./subtask-1a--bulgarian/dataset_train_v1_bulgarian.tsv', './subtask-1a--bulgarian/folds/')
