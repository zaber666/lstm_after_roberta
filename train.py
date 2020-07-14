import config
from utils import *
from models import TokenModel, CharacterModel
from datasets import TokenDataset, CharacterDataset
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
import argparse
import os

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def train_token_model(model, device, dataloaders_dict, criterion, optimizer, fold):


    best_jaccard = 0.0

    for epoch in range(config.token_num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            tk0 = tqdm(dataloaders_dict[phase], total=len(dataloaders_dict[phase]) )
            for i, data in enumerate(tk0):
                ids = data['ids'].to(device)
                masks = data['masks'].to(device)
                tweet = data['tweet']
                selected_text = data['selected_text']
                offsets = data['offsets'].numpy()
                start_idx = data['start_idx'].to(device)
                end_idx = data['end_idx'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    start_logits, end_logits = model(ids, masks)

                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()


                    epoch_loss += loss.item() * len(ids)

                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            selected_text[i],
                            start_logits[i],
                            end_logits[i])

                        epoch_jaccard += jaccard_score


            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, config.token_num_epochs, phase, epoch_loss, epoch_jaccard))

        if best_jaccard < epoch_jaccard:
            file_name = "token_model_fold_" + str(fold) + ".pth"
            if not os.path.exists('weights/'):
                os.makedirs('weights/')
            torch.save(model.state_dict(), 'weights/'+file_name )
            best_jaccard = epoch_jaccard







def list_for_charater_model(model, device, dataloader):
    texts = []
    selected_texts = []
    offsets_list = []
    start_logits_list = []
    end_logits_list = []

    tk0 = tqdm(dataloader, total=len(dataloader))

    for idx, data in enumerate(tk0):
        ids = data['ids'].to(device)
        masks = data['masks'].to(device)
        tweet = data['tweet']
        offsets = data['offsets'].numpy()
        start_idx = data['start_idx'].to(device)
        end_idx = data['end_idx'].to(device)
        selected_text = data['selected_text']


        with torch.no_grad():
            start_logits, end_logits = model(ids, masks)

        start_idx = start_idx.cpu().detach().numpy()
        end_idx = end_idx.cpu().detach().numpy()
        start_logits = start_logits.cpu().detach().numpy()
        end_logits = end_logits.cpu().detach().numpy()


        for i in range(len(ids)):

            texts.append(tweet[i])
            selected_texts.append(selected_text[i])
            offsets_list.append(offsets[i])
            start_logits_list.append(start_logits[i,:])
            end_logits_list.append(end_logits[i,:])
    return texts, selected_texts, offsets_list, start_logits_list, end_logits_list




def train_character_model(model, device, dataloaders_dict, criterion, optimizer, fold):
    best_jaccard = 0.0

    for epoch in range(config.char_num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            epoch_loss = 0.0
            epoch_jaccard = 0.0
            tk0 = tqdm(dataloaders_dict[phase], total=len(dataloaders_dict[phase]) )
            
            for i, data in enumerate(tk0):
                char_ids = data['char_id']
                start_logits = data['start_logits']
                end_logits = data['end_logits']
                start_idx = data['start_idx'].to(device)
                end_idx = data['end_idx'].to(device)
                tweet = data['text']
                selected_text = data['selected_text']
                

                bs = len(char_ids)

                inp = torch.stack([char_ids, start_logits, end_logits], dim=-1)
                inp = torch.reshape(inp, (bs,156,3)).to(device)

                h_0 = np.zeros((config.lstm_num_layers*config.lstm_direction, bs, config.lstm_hidden_size))
                c_0 = np.zeros((config.lstm_num_layers*config.lstm_direction, bs, config.lstm_hidden_size))
                h_0 = torch.from_numpy(h_0).float().to(device)
                c_0 = torch.from_numpy(c_0).float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    
                    start_logits2, end_logits2 = model(inp, h_0, c_0)
                    loss = criterion(start_logits2, end_logits2, start_idx, end_idx)
                    
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()


                    epoch_loss += loss.item() * len(char_ids)

                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits2 = torch.softmax(start_logits2, dim=1).cpu().detach().numpy()
                    end_logits2 = torch.softmax(end_logits2, dim=1).cpu().detach().numpy()

                    for i in range(len(char_ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            selected_text[i],
                            start_logits2[i],
                            end_logits2[i])
                        epoch_jaccard += jaccard_score



            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, config.char_num_epochs, phase, epoch_loss, epoch_jaccard))

        if epoch_jaccard > best_jaccard:
            file_name = 'character_model_fold_' +str(fold) + ".pth"
            torch.save(model.state_dict(), 'weights/'+file_name)
            best_jaccard = epoch_jaccard




if __name__ == "__main__":

    seed_everything(42)

    train_df = pd.read_csv(config.training_file)
    train_df.dropna(inplace=True)
    train_df.text = train_df.text.astype(str)
    train_df.selected_text = train_df.selected_text.astype(str)

    for fold in range(5):
        df_train = train_df[train_df.kfold != fold].reset_index(drop=True)
        df_val = train_df[train_df.kfold == fold].reset_index(drop=True)
        

        token_train_loader = torch.utils.data.DataLoader(
            dataset=TokenDataset(df_train),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

        token_val_loader = torch.utils.data.DataLoader(
            dataset=TokenDataset(df_val),
            batch_size=config.batch_size,
            num_workers=2,
            drop_last=True
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        token_model = TokenModel()
        token_model.to(device)
        dataloaders_dict = {'train':token_train_loader, 'val':token_val_loader}
        optimizer = optim.AdamW(token_model.parameters(), lr=config.token_lr, betas=(0.9, 0.999))
        criterion = loss_fn
        print('Training Roberta for fold ' + str(fold))
        train_token_model(token_model, device, dataloaders_dict, criterion, optimizer, fold)


        # Token model has been trained. Now we will use the best model to proceed for the character model.

        token_model.load_state_dict(torch.load('weights/token_model_fold_' + str(fold) + ".pth"))

        print('Perparing Character Model Data.')
        train_texts, train_selected_texts, train_offsets, train_start_logits_list, train_end_logits_list = list_for_charater_model(token_model, device, token_train_loader)
        val_texts, val_selected_texts, val_offsets_list, val_start_logits_list, val_end_logits_list = list_for_charater_model(token_model, device, token_val_loader)

        char_train_dataset = CharacterDataset(train_texts, train_selected_texts, train_offsets, train_start_logits_list, train_end_logits_list)
        char_val_dataset = CharacterDataset(val_texts, val_selected_texts, val_offsets_list, val_start_logits_list, val_end_logits_list)

        char_train_loader = torch.utils.data.DataLoader(dataset=char_train_dataset, batch_size=config.batch_size, shuffle=False)
        char_val_loader = torch.utils.data.DataLoader(dataset=char_val_dataset, batch_size=config.batch_size, shuffle=False)

        dataloaders_dict = {'train':char_train_loader, 'val':char_val_loader}

        char_model = CharacterModel()
        char_model.to(device)
        
        optimizer = optim.AdamW(char_model.parameters(), lr=config.character_lr, betas=(0.9, 0.999))
        print('Training Character Model for fold ' + str(fold))
        train_character_model(char_model, device, dataloaders_dict, criterion, optimizer, fold)
