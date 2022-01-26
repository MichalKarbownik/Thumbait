import re
from typing import Union
import pickle

import torch
import fasttext
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

from logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    def __init__(
        self,
        _type: str = "big",
        fasttext_path="models/cc.en.300.bin",
    ):
        """ """
        if _type == "big":
            self.model = fasttext.load_model(fasttext_path)
        elif _type == "light":
            with open("data/vectorizer.pckl", "rb") as file:
                self.vectorizer = pickle.load(file)

        self.en_stop = stopwords.words("english")
        self.stemmer = WordNetLemmatizer()
        self._type = _type

    def _preprocess_text(
        self,
        document,
    ):
        # Remove all the special characters
        document = re.sub(r"\W", " ", str(document))

        # remove all single characters
        document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)

        # Remove single characters from the start
        document = re.sub(r"\^[a-zA-Z]\s+", " ", document)

        # Substituting multiple spaces with single space
        document = re.sub(r"\s+", " ", document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r"^b\s+", "", document)

        # Converting to Lowercase
        document = document.lower()

        if self._type == "light":
            return document

        # Lemmatization
        tokens = document.split()
        tokens = [self.stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in self.en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        # preprocessed_text = ' '.join(tokens)
        preprocessed_text = tokens

        return preprocessed_text

    def _pad_collate(self, documents: Union[list, torch.Tensor]):
        texts = [text if len(text) else torch.zeros(1, 300) for text in documents]
        text_lens = [len(text) for text in texts]
        texts_pad = pad_sequence(texts, batch_first=True, padding_value=0)
        texts_pad = pack_padded_sequence(
            texts_pad, text_lens, batch_first=True, enforce_sorted=False
        )
        return texts_pad

    def preprocess_texts(self, documents: list) -> PackedSequence:
        """
        Return preprocessed documents
        """

        if self._type == "big":
            tensors = self._pad_collate(
                [
                    torch.tensor(
                        np.array(
                            [
                                self.model[word]
                                for word in self._preprocess_text(document)
                            ]
                        )
                    )
                    for document in documents
                ]
            )
            return tensors

        elif self._type == "light":
            return torch.tensor(
                self.vectorizer.transform(
                    [self._preprocess_text(document) for document in documents]
                )
                .toarray()
                .astype("float32")
            )
