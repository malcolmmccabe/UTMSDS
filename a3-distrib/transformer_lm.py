# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import * 


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, d_model, num_positions, d_internal, num_classes, num_layers):
        super().__init__()
        self.transformer = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def set_vocab_index(self, vocab_index):
        """
        Sets the vocabulary indexer. This is an extra method if you need to store the vocab index inside the class.
        """
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        self.transformer.eval()
        for c in context:
            indexed_context = []
            indexed_context.append(self.vocab_index.index_of(c))
        context_tensor = torch.LongTensor(indexed_context).unsqueeze(0)
    
        with torch.no_grad():
            log_probs, _ = self.transformer(context_tensor)
            return log_probs[0, -1].cpu().numpy()
        

    def get_log_prob_sequence(self, next_chars, context):
        self.transformer.eval()
        full_sequence = context + next_chars
        indexed_context = []
        for c in context:
            indexed_context.append(self.vocab_index.index_of(c))
        
        sequence_tensor = torch.LongTensor(indexed_context).unsqueeze(0)
        with torch.no_grad():
            log_probs, _ = self.transformer(sequence_tensor)
            total_log_prob = 0.0
            for i, char in enumerate(next_chars):
                char_idx = self.vocab_index.index_of(char)
                total_log_prob += log_probs[0, len(context) + i, char_idx].item()
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    vocab_size = 27
    d_model = 128
    d_internal = 256
    num_positions = 20
    num_layers = 2
    lr = 1e-4
    num_epochs = 10
    num_classes = 3

    model = NeuralLanguageModel(vocab_size, d_model, num_positions, d_internal, num_classes, num_layers)
    optimizer = optim.Adam(model.transformer.parameters(), lr=lr)
    loss_fcn = nn.CrossEntropyLoss()

    train_sequences = []
    for i in range(len(train_text) - num_positions):
        sequence = train_text[i:i + num_positions + 1]
        train_sequences.append(sequence)
    
    dev_sequences = []
    for i in range(len(dev_text) - num_positions):
        sequence = dev_text[i+i + num_positions + 1]
        dev_sequences.append(sequence)

    for epoch in range(num_epochs):
        model.transformer.train()
        total_loss = 0.0

        for seq in train_sequences:

            input_indices = []
            target_indices = []

            for c in seq[:-1]:
                input_indices.append(vocab_index.index_of(c))
            
            for c in seq[:-1]:
                target_indices.append(vocab_index.index_of(c))

            input_seq = torch.LongTensor(input_indices).unsqueeze(0)
            target_seq = torch.LongTensor(target_indices).unsqueeze(0)

            log_probs, _ = model.transformer(input_seq)
            

            log_probs = log_probs.view(-1, vocab_size) 
            target_seq = target_seq.view(-1)  

            loss = loss_fcn(log_probs, target_seq)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.item()

    return model