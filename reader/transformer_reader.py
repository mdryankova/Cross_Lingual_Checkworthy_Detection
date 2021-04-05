import torch
import pandas as pd
from transformers import AutoTokenizer
from config import LANGS_IDS
import re


def preprocess(text):
    text = re.sub(r"https?://(S+)", "[url]", text)
    return text


class CheckWorthyDetectionTask(torch.utils.data.Dataset):
    def __init__(self, file_path, lang, tokenizer_name, max_seq_len):
        self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len
        self.special_tokens_dict = {'additional_special_tokens': ["[user]"]}
        self.tokenizer.add_special_tokens(self.special_tokens_dict)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        tweet_text = row['tweet_text']
        topic_id = row['topic_id']
        tweet_id = row['tweet_id']
        label = row['check_worthiness']
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

    def __len__(self):
        return self.data.shape[0]
