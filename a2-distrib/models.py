# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import nltk


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(nn.Module, SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, word_embeddings, hid, num_classes, out):

        super(NeuralSentimentClassifier, self).__init__()
        self.word_embeddings = word_embeddings
        self.embedding_layer = self.word_embeddings.get_initialized_embedding_layer()
        self.embedding_dim = self.word_embeddings.get_embedding_length()
        self.dropout = nn.Dropout(p=out)

        self.V = nn.Linear(self.embedding_dim, hid)
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        embeddings = self.embedding_layer(x)
        averaged_embedding = torch.mean(embeddings, dim=1)
        hidden_output = self.V(averaged_embedding)
        hidden_output = self.g(hidden_output) 
        hidden_output = self.dropout(hidden_output)
        output = self.W(hidden_output)
        log_probs = self.log_softmax(output) 
        return log_probs

    def correct_spelling(self, word, word_indexer, prefix_length=3):
        
        if word_indexer.index_of(word) != -1:
            return word
        
        closest_word = None
        min_distance = float('inf')


        options = [word_indexer.get_object(i) for i in range(len(word_indexer)) 
               if word_indexer.get_object(i).startswith(word[:prefix_length])]

        if not options:
            options = [word_indexer.get_object(i) for i in range(len(word_indexer))]

        for vocab_word in options:
            if vocab_word == 'UNK':
                continue
            dist = nltk.edit_distance(word, vocab_word)
            if dist < min_distance:
                min_distance = dist
                closest_word = vocab_word
        
        return closest_word if closest_word is not None else 'UNK'
    
    def predict(self, ex_words, has_typos):

        word_indices = []
        for word in ex_words:
            if has_typos:
                corrected_word = self.correct_spelling(word, self.word_embeddings.word_indexer)
            
            else:
                corrected_word = word

        word_idx = self.word_embeddings.word_indexer.index_of(corrected_word)
        if word_idx == -1:
            word_idx = self.word_embeddings.word_indexer.index_of("UNK")  # Handle unknown words
        word_indices.append(word_idx)

        word_indices = torch.tensor(word_indices, dtype=torch.long).unsqueeze(0)

        log_probs = self.forward(word_indices)
        predicted_class = torch.argmax(log_probs, dim=1).item()
        return predicted_class



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    
    hid = 128
    num_classes = 2
    out = 0.5
    classifier = NeuralSentimentClassifier(word_embeddings, hid, num_classes, out)
    num_epochs = 5

    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    loss_function = nn.NLLLoss()    

    for epoch in range(0, num_epochs):
        total_loss = 0
        random.shuffle(train_exs)
        for ex in train_exs:
            words, label = ex.words, ex.label
            word_indices = [word_embeddings.word_indexer.index_of(word) if word_embeddings.word_indexer.index_of(word) != -1 else word_embeddings.word_indexer.index_of("UNK") for word in words]


            
            word_indices = torch.tensor(word_indices, dtype=torch.long)
            
            word_indices = word_indices.unsqueeze(0)


            classifier.zero_grad()

            log_probs = classifier.forward(word_indices)
            loss = loss_function(log_probs, torch.tensor([label], dtype=torch.long))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return classifier