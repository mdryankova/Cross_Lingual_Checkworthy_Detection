import pandas as pd


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


if __name__ == '__main__':
    normalize_turkish_dataset('./subtask-1a--turkish/dataset_train_v1_turkish.tsv',
                              './subtask-1a--turkish/normalized/dataset_train_v1_turkish.tsv')
    normalize_turkish_dataset('./subtask-1a--turkish/dataset_dev_v1_turkish.tsv',
                              './subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv')

    normalize_spanish_dataset('./subtask-1a--spanish/dataset_train.tsv',
                              './subtask-1a--spanish/normalized/dataset_train.tsv')
    normalize_spanish_dataset('./subtask-1a--spanish/dataset_dev.tsv',
                              './subtask-1a--spanish/normalized/dataset_dev.tsv')

    normalize_english_dataset('./subtask-1a--english/dataset_train_v1_english.tsv',
                              './subtask-1a--english/normalized/dataset_train.tsv')
    normalize_english_dataset('./subtask-1a--english/dataset_dev_v1_english.tsv',
                              './subtask-1a--english/normalized/dataset_dev.tsv')
