import numpy as np
import torch
import torch.nn as nn


def strip_unwanted(offsets, start_logits, end_logits):
    act_offsets = []
    act_s_log = []
    act_e_log = []

    for i in range(len(offsets)):
        offset = offsets[i]
        end_idx = -1
        for j in range(96 - 4):
            if offset[j+4][0] == 0 and offset[j+4][1] == 0:
                end_idx = j+4
                break
        act_offsets.append(offset[4:end_idx])
        act_s_log.append(start_logits[i][4:end_idx])
        act_e_log.append(end_logits[i][4:end_idx])
    return act_offsets, act_s_log, act_e_log


def character_dataset_preprocess(text, start_logits, end_logits, offsets, dictionary):
    maxLen = 156

    chars = np.zeros(maxLen)
    for i, alph in enumerate(text[1:]):
        if alph in dictionary.keys():
            chars[i] = dictionary[alph]
        else:
            chars[i] = 77

    char_start_logits = np.zeros(len(text) - 1)

    for i, prob in enumerate(start_logits):
        offset1 = offsets[i][0]
        offset2 = offsets[i][1]
        if i == 0:
            char_start_logits[ : offset2-1] = prob
        else:
            char_start_logits[offset1 - 1 : offset2 - 1] = prob

    char_end_logits = np.zeros(len(text) - 1)

    for i, prob in enumerate(end_logits):
        offset1 = offsets[i][0]
        offset2 = offsets[i][1]
        if i == 0:
            char_end_logits[ : offset2-1] = prob
        else:
            char_end_logits[offset1 - 1 : offset2 - 1] = prob

    n = len(char_start_logits)

    if n < maxLen:



        char_start_logits_pad = np.zeros(maxLen, dtype=float)
        for j in range(n):
            char_start_logits_pad[j] = char_start_logits[j]

        char_end_logits_pad = np.zeros(maxLen, dtype=float)
        for j in range(n):
            char_end_logits_pad[j] = char_end_logits[j]

    return chars, char_start_logits_pad, char_end_logits_pad

def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(text, selected_text, start_logits, end_logits):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = text[start_pred:end_pred]
    
    return jaccard(selected_text, pred)

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss