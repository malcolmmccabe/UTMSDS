# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import nltk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader



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

        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """

        super(NeuralSentimentClassifier, self).__init__()
        self.word_embeddings = word_embeddings
        self.embedding_layer = self.word_embeddings.get_initialized_embedding_layer(frozen = False, padding_idx = 0)
        self.embedding_dim = self.word_embeddings.get_embedding_length()
        self.dropout = nn.Dropout(p=out)

        self.V = nn.Linear(self.embedding_dim, hid)
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    
    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        return self.log_softmax(self.W(self.dropout(self.g(self.V(torch.mean(self.embedding_layer(x), dim=1))))))

    def correct_spelling(self, word, word_indexer, prefix_length=3):
        
        # words in index
        if word_indexer.index_of(word) != -1:
            return word
        
        # Instantiate variables
        closest_word = None
        min_distance = float('inf')

        options = []
        for i in range(len(word_indexer)): 
            new_word = word_indexer.get_object(i)

            if new_word.startswith(word[:prefix_length]):
                options.append(new_word)

        if not options:
            options = []
            for i in range(len(word_indexer)):
                new_word = word_indexer.get_object(i)
                options.append(new_word)


        for new_word in options:
            if new_word == 'UNK':
                continue
            dist = nltk.edit_distance(word, new_word)
            if dist < min_distance:
                min_distance = dist
                closest_word = new_word
        
        return closest_word if closest_word is not None else 'UNK'
    
    def predict(self, ex_words, has_typos):

        if has_typos: 
            corrected_words = []
            for word in ex_words:
                corrected_word = self.correct_spelling(word, self.word_embeddings.word_indexer)
                corrected_words.append(corrected_word)
        else:
            corrected_words = ex_words

        word_indices = []
        for word in corrected_words:
            word_index = self.word_embeddings.word_indexer.index_of(word)
            if word_index == -1:
                word_index = self.word_embeddings.word_indexer.index_of('UNK')
            word_indices.append(word_index)

    
        word_indices = torch.tensor(word_indices, dtype=torch.long).unsqueeze(0)

        log_probs = self.forward(word_indices)
        predicted_class = torch.argmax(log_probs, dim=1).item()
        return predicted_class


def batching(batch, word_indexer, pad_idx=0):
    
    index_sentences = []

    for sentence in batch:
        sentence_indices = []
        for word in sentence:
            word_index = word_indexer.index_of(word)
            if word_index == -1:
                word_index = word_indexer.index_of('UNK')
            sentence_indices.append(word_index)
    
        index_sentences.append(torch.tensor(sentence_indices))
    
    return pad_sequence(index_sentences, batch_first=True, padding_value=pad_idx)

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
    # Constants
    hid = 128
    num_classes = 2
    out = 0.5
    classifier = NeuralSentimentClassifier(word_embeddings, hid, num_classes, out)
    #num_epochs = 10
    batch_size = 32

    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    loss_function = nn.NLLLoss()   

    def batching_transform(batch):

        sentences = []
        for ex in batch:
            sentences.append(ex.words)
        
        labels = []
        for ex in batch:
            labels.append(ex.label)

        padded_sentences = batching(sentences, word_embeddings.word_indexer)
        return padded_sentences, torch.tensor(labels)

    train_loader = DataLoader(train_exs, batch_size=batch_size, shuffle=True, collate_fn=batching_transform)

    # Loop through epochs (specified in parameters)
    for epoch in range(0, args.num_epochs):
        total_loss = 0
        
        for batch_sentences, batch_labels in train_loader:
            classifier.zero_grad()

            # Forward
            log_probs = classifier.forward(batch_sentences)

            # Loss
            loss = loss_function(log_probs, batch_labels)

            #Backwards
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return classifier