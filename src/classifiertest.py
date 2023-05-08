from typing import List
import pandas as pd
import torch
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import AdamW
from torch.optim import SGD, RMSprop, Adam
from transformers import get_linear_schedule_with_warmup
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import tqdm
from copy import deepcopy
from models import RoBERTaDataset
from sklearn.model_selection import train_test_split



class Classifier:

    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(self, config = None):
      if config is not None:
          self.config = config
      else:
          self.config = {
              'max_len': 128,
              'batch_size': 16,
              'epoch': 30,
              'lr': 5e-5
          }



    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # basic settings
        ## loading the files
        columns = ['Polarity', 'Aspect Category', 'target term', 'character offsets', 'sentence']
        df_train = pd.read_csv(train_filename, sep='\t', header=None, names=columns)
        df_dev = pd.read_csv(dev_filename, sep='\t', header=None, names=columns)
        ## preprocessing
        df_train['sentence'] = df_train['sentence'].str.lower()
        df_dev['sentence'] = df_dev['sentence'].str.lower()
        polarity_mapping = {"positive": 1, "negative": 2, "neutral": 0}

        # eval function
        def eval_list(glabels, slabels):
            if (len(glabels) != len(slabels)):
                print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
                len(slabels), len(glabels)))
            n = min(len(slabels), len(glabels))
            incorrect_count = 0
            for i in range(n):
                if slabels[i] != glabels[i]: incorrect_count += 1
            acc = (n - incorrect_count) / n
            return acc*100


        # loading train and dev
        ## Load the ROBERTa tokenizer
        #tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        ## Set the maximum length for tokenized input
        max_len = self.config['max_len']
        ## Create training and testing datasets
        train_dataset = RoBERTaDataset(df_train, tokenizer, max_len)
        test_dataset = RoBERTaDataset(df_dev, tokenizer, max_len)
        ## Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)


        # model
        ## Load the RoBERTa model for sequence classification
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
        ## Use the correct device
        model.to(device)


        ####### TRAIN #######

        # training settings
        ## epochs
        epochs = self.config['epoch']
        ## optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'])
        ## criterion
        criterion = nn.CrossEntropyLoss()
        ## scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # training
        for epoch in range(epochs):
            model.train()  # Set the model to training mode
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # Convert polarity labels to numeric values
                labels = [polarity_mapping[label] for label in batch['polarity']]
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Calculate loss
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


        ####### VAL #######

        # Put the model in evaluation mode
        model.eval()

        # Initialize variables to store predictions and labels
        all_predictions = []
        all_labels = []

        # Iterate over test_dataloader
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = [polarity_mapping[label] for label in batch['polarity']]
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            with torch.no_grad():
                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits

                # Compute predicted labels
                predictions = torch.argmax(logits, dim=1)

                # Append predictions and labels to the lists
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        # Convert lists to tensors
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)

        # Calculate accuracy
        # accuracy = calculate_accuracy(all_predictions, all_labels)
        accuracy = eval_list(all_predictions, all_labels)


        # saving the model
        self.final_model = deepcopy(model.state_dict())

        return accuracy



    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        # Basic settings
        ## Load the file
        columns = ['Polarity', 'Aspect Category', 'target term', 'character offsets', 'sentence']
        df_data = pd.read_csv(data_filename, sep='\t', header=None, names=columns)
        ## preprocessing
        df_data['sentence'] = df_data['sentence'].str.lower()
        polarity_mapping = {"positive": 1, "negative": 2, "neutral": 0}


        # eval function
        def eval_list(glabels, slabels):
            if (len(glabels) != len(slabels)):
                print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
                len(slabels), len(glabels)))
            n = min(len(slabels), len(glabels))
            incorrect_count = 0
            for i in range(n):
                if slabels[i] != glabels[i]: incorrect_count += 1
            acc = (n - incorrect_count) / n
            return acc*100

        # loading data
        ## Load the ROBERTa tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        ## Set the maximum length for tokenized input
        max_len = self.config['max_len']
        ## Create training and testing datasets
        dataset = RoBERTaDataset(df_data, tokenizer, max_len)
        ## Create data loaders
        data_loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

        # Model
        ## Load the RoBERTa model for sequence classification
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
        ## Load model
        model.load_state_dict(self.final_model)
        ## Use the correct device
        model.to(device)


        ####### PREDICT #######
        model.eval()
        # Initialize variables to store predictions and labels
        all_predictions = []
        all_labels = []

        # Iterate over test_dataloader
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = [polarity_mapping[label] for label in batch['polarity']]
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            with torch.no_grad():
                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits

                # Compute predicted labels
                predictions = torch.argmax(logits, dim=1)

                # Append predictions and labels to the lists
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        # Convert lists to tensors
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)

        # Calculate accuracy
        # accuracy = calculate_accuracy(all_predictions, all_labels)
        accuracy = eval_list(all_predictions, all_labels)

        return all_predictions.numpy()

