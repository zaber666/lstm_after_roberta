import config
import torch
import tokenizers
import numpy as np
from utils import *
import json


# The text is tokenized using roberta-base vocabulary. 
# RoBERTa model for Question Answering expects the input of the model be in form: [cls_token] [Tokes ids of the question] [sep_token] [......Token ids of the text corpus.......] [sep_token]
# For roberta cls_token --> 0 and sep_token --> 2
# max_len = 96


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=config.vocab_file, 
            merges_file=config.merges_file, 
            lowercase=True,
            add_prefix_space=True)

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        
        ids, masks, tweet, offsets = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        data['selected_text'] = row.selected_text.lower()
        
        
        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
        
        return data

    def __len__(self):
        return len(self.df)
    
    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
                
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
        
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        
        return ids, masks, tweet, offsets


    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " +  " ".join(row.selected_text.lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        try:
            start_idx = target_idx[0]
            end_idx = target_idx[-1]
        except:
            print("Error while finding start_idx and end_idx.")
            
        
        return start_idx, end_idx



# Distributed the logits of a token to every chacater in it.
# max_len = 156

class CharacterDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, selected_text_list, offset_list, start_logits_list, end_logits_list):
        self.texts = text_list
        self.selected_texts = selected_text_list
        self.offsets, self.start_logits, self.end_logits = strip_unwanted(torch.tensor(offset_list).numpy(), np.array(start_logits_list), np.array(end_logits_list))
    
    
        with open(config.character_dictionary, 'r') as f:
            self.dictionary = json.load(f)
    
    def __len__(self):
        return len(self.texts)
  
    def __getitem__(self, index):

        data = {}
        data['text'] = self.texts[index][1:]
        selected_text = " " +  " ".join(self.selected_texts[index].lower().split())
        data['selected_text'] = selected_text[1:]
        char_id, start_log, end_log = character_dataset_preprocess(self.texts[index], self.start_logits[index], self.end_logits[index], self.offsets[index], self.dictionary)
        data['char_id'] = torch.from_numpy(char_id).float()
        data['start_logits'] = torch.from_numpy(start_log).float()
        data['end_logits'] = torch.from_numpy(end_log).float()
        data['start_idx'] = data['text'].find(data['selected_text'])
        data['end_idx'] = data['start_idx'] + len(data['selected_text'])

        return data











