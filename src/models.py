import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.optim import SGD, RMSprop, Adam
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from typing import List
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet



class RoBERTaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Split the offset into start and end indices
        start, end = map(int, self.df.iloc[index]['character offsets'].split(':'))
        sentence = self.df.iloc[index]['sentence']
        aspect_category = self.df.iloc[index]['Aspect Category']
        new_sentence = sentence[:start] + aspect_category + sentence[end:]
        # Replace the target word in the sentence with the aspect category multiple times, if needed
        num_replacements = end - start - len(aspect_category)
        for _ in range(num_replacements):
            new_sentence = new_sentence[:start] + aspect_category + new_sentence[start + len(aspect_category):]

        offset_vector = [1 if start <= i < end else 0 for i in range(len(sentence))]
        target_term = self.df.iloc[index]['target term']
        polarity = self.df.iloc[index]['Polarity']
        # Convert polarity label to numeric value
        # polarity = polarity_mapping[polarity]

        # Combine sentence and target term
        text = f"{sentence} [SEP] {aspect_category} [SEP] {target_term}"
# [SEP] {offset_vector}
        # Tokenize the text
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True # enable truncation
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'polarity': polarity
        }
