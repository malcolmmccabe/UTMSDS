# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import string


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')

class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

            entailment_score = torch.softmax(logits, dim=1)[0][0].item()

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        return entailment_score

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):

    def __init__(self, threshold_s =0.12, threshold_ns = 0.09, jaccard_weight = 0.64):
        # Set self variables
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.threshold_s = threshold_s
        self.threshold_ns = threshold_ns
        self.jaccard_weight = jaccard_weight

    def preprocess(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = text.split()
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    
    def jaccard_similarity(self, fact, passage):
        # Calculate Jaccard Similarity
        fact_set = set(fact.split())
        passage_set = set(passage.split())
        intersection = fact_set.intersection(passage_set)
        union = fact_set.union(passage_set)
        if union:
            return len(intersection) / len(union)
        
    def token_overlap_similarity(self, fact, passage):
        
        fact_set = set(fact.split())
        passage_set = set(passage.split())
        intersection = fact_set.intersection(passage_set)
        if fact_set:
            return len(intersection) / len(fact_set)
        else:
            return 0
    
    def predict(self, fact: str, passages: list) -> str:
        fact = self.preprocess(fact)
        max_combined_similarity = 0
        
        for passage in passages:
            passage_text = self.preprocess(passage['text'])
            
            # Use both jaccard and overlap
            jaccard_sim = self.jaccard_similarity(fact, passage_text)
            token_overlap_sim = self.token_overlap_similarity(fact, passage_text)
            
            # Combine both methods
            combined_similarity = (self.jaccard_weight * jaccard_sim) + ((1 - self.jaccard_weight) * token_overlap_sim)
            combined_similarity /= (1 + len(passage_text.split()) / 100)
            
            max_combined_similarity = max(max_combined_similarity, combined_similarity)
        
        # Threshold define
        if max_combined_similarity >= self.threshold_s:
            return "S"
        elif max_combined_similarity < self.threshold_ns:
            return "NS"
        
        return "NS"

class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        for passage in passages:
            sentences = passage['text'].split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                entailment_score = self.ent_model.check_entailment(sentence, fact)

                # If meets treshold, S
                if entailment_score >= 0.5:
                    return "S"
        # If under threshold, NS
        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

