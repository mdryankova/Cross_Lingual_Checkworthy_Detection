RANDOM_SEED = 0

DATASET_PATHS = {
    'english': {
        'train': 'data/subtask-1a--english/normalized/dataset_train.tsv',
        'dev': 'data/subtask-1a--english/normalized/dataset_dev.tsv',
        'test': 'data/subtask-1a--english/dataset_test_input_english.tsv'
    },
    'turkish': {
        'train': 'data/subtask-1a--turkish/normalized/dataset_train_v1_turkish.tsv',
        'dev': 'data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv',
        'test': 'data/subtask-1a--turkish/dataset_test_v1_turkish.tsv',
    },
    'spanish': {
        'train': 'data/subtask-1a--spanish/normalized/dataset_train.tsv',
        'dev': 'data/subtask-1a--spanish/normalized/dataset_dev.tsv',
        'test': 'data/subtask-1a--spanish/dataset_test_participants.tsv',
    },
    'bulgarian': {
        'train': 'data/subtask-1a--bulgarian/dataset_train_v1_bulgarian.tsv',
        'dev': 'data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv',
        'test': 'data/subtask-1a--bulgarian/dataset_test_input_bulgarian.tsv',
    },
    'arabic': {
        'train': 'data/subtask-1a--arabic/normalized/train.tsv',
        'dev': 'data/subtask-1a--arabic/normalized/dev.tsv',
        'test': 'data/subtask-1a--arabic/CT21-AR-Test-T1.tsv',
    }
}

CV_DATASET_PATHS = {
    'english': {
        'train': 'data/subtask-1a--english/folds',
        'dev': 'data/subtask-1a--english/normalized/dataset_dev.tsv',
        'test': 'data/subtask-1a--english/dataset_test_input_english.tsv'
    },
    'turkish': {
        'train': 'data/subtask-1a--turkish/folds',
        'dev': 'data/subtask-1a--turkish/normalized/dataset_dev_v1_turkish.tsv',
        'test': 'data/subtask-1a--turkish/dataset_test_v1_turkish.tsv',
    },
    'spanish': {
        'train': 'data/subtask-1a--spanish/folds',
        'dev': 'data/subtask-1a--spanish/normalized/dataset_dev.tsv',
        'test': 'data/subtask-1a--spanish/dataset_test_participants.tsv',
    },
    'bulgarian': {
        'train': 'data/subtask-1a--bulgarian/folds',
        'dev': 'data/subtask-1a--bulgarian/dataset_dev_v1_bulgarian.tsv',
        'test': 'data/subtask-1a--bulgarian/dataset_test_input_bulgarian.tsv',
    },
    'arabic': {
        'train': 'data/subtask-1a--arabic/folds',
        'dev': 'data/subtask-1a--arabic/normalized/dev.tsv',
        'test': 'data/subtask-1a--arabic/CT21-AR-Test-T1.tsv',
    }
}

LANGS_IDS = {
    'english': 0,
    'turkish': 1,
    'spanish': 2,
    'bulgarian': 3,
    'arabic': 4
}
