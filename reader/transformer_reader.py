import emoji
import torch
import pandas as pd
from transformers import AutoTokenizer
from config import LANGS_IDS
import re


def preprocess(text):
    text = re.sub(r"https?://(S+)", "[url]", text)
    return text


class CheckWorthyDetectionTask(torch.utils.data.Dataset):
    def __init__(self, file_path, lang, tokenizer_name, max_seq_len, preprocess=None):
        self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len
        self.preprocess = preprocess
        if self.preprocess:
            special_tokens_list = list(item for item in emoji.UNICODE_EMOJI['en'].keys()) + ['[url]']
            self.special_tokens_dict = {'additional_special_tokens': special_tokens_list}
            self.tokenizer.add_special_tokens(self.special_tokens_dict)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        if self.preprocess:
            tweet_text = preprocess(row['tweet_text'])
        else:
            tweet_text = row['tweet_text']  # preprocess(row['tweet_text'])
        topic_id = row['topic_id']
        tweet_id = row['tweet_id']
        encoding = self.tokenizer.encode_plus(tweet_text,
                                              add_special_tokens=True,
                                              max_length=self.max_seq_len,
                                              padding='max_length',  # Pad & truncate all sentences.
                                              truncation=True,
                                              return_token_type_ids=False,
                                              return_attention_mask=True,  # Construct attn. masks.
                                              return_tensors='pt'  # Return pytorch tensors.
                                              )
        encoded_lang = LANGS_IDS[self.lang]
        if 'check_worthiness' in row:
            label=row['check_worthiness']
            return dict(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                label=torch.LongTensor([label]),
                tweet_text=tweet_text,
                topic_id=topic_id,
                lang=self.lang,
                tweet_id=tweet_id,
                encoded_langs=torch.LongTensor([encoded_lang]),
            )
        else:
            return dict(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                tweet_text=tweet_text,
                topic_id=topic_id,
                lang=self.lang,
                tweet_id=tweet_id,
                encoded_langs=torch.LongTensor([encoded_lang]),
            )

    def __len__(self):
        return self.data.shape[0]
