RANDOM_SEED = 0

DATASET_PATHS = {
    'english': {
        'train': 'data/subtask-1a--english/normalized/dataset_train.tsv',
        'dev': 'data/subtask-1a--english/normalized/dataset_dev.tsv',
        'test': None
    },
    'turkish': {
        'train': 'data/subtask-1a--turkish/normalized/dataset_train_v1_turkish.tsv',
        'dev': 'data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv',
        'test': None,
    },
    'spanish': {
        'train': 'data/subtask-1a--spanish/normalized/dataset_train.tsv',
        'dev': 'data/subtask-1a--spanish/normalized/dataset_dev.tsv',
        'test': None,
    },
    'bulgarian': {
        'train': 'data/subtask-1a--bulgarian/dataset_train_v1_bulgarian.tsv',
        'dev': 'data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv',
        'test': None,
    },
    'arabic': {
        'train': 'data/subtask-1a--arabic/normalized/dataset_train.tsv',
        'dev': 'data/subtask-1a--arabic/normalized/dataset_dev.tsv',
        'test': None,
    }
}

CV_DATASET_PATHS = {
    'english': {
        'train': 'data/subtask-1a--english/folds',
        'dev': 'data/subtask-1a--english/normalized/dataset_dev.tsv',
        'test': None
    },
    'turkish': {
        'train': 'data/subtask-1a--turkish/folds',
        'dev': 'data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv',
        'test': None,
    },
    'spanish': {
        'train': 'data/subtask-1a--spanish/folds',
        'dev': 'data/subtask-1a--spanish/normalized/dataset_dev.tsv',
        'test': None,
    },
    'bulgarian': {
        'train': 'data/subtask-1a--bulgarian/folds',
        'dev': 'data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv',
        'test': None,
    },
    'arabic': {
        'train': 'data/subtask-1a--arabic/folds',
        'dev': 'data/subtask-1a--arabic/normalized/dataset_dev.tsv',
        'test': None,
    }
}

LANGS_IDS = {
    'english': 0,
    'turkish': 1,
    'spanish': 2,
    'bulgarian': 3,
    'arabic': 4
}
