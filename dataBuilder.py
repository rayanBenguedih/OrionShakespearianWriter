import tensorflow as tf
import numpy as np
import os
import time


class dataTokenizerBuilder():
    def __init__(self,filename, path_to_file, seq_length=100):
        self.path_to_file = tf.keras.utils.get_file(filename, path_to_file)
        self.text = self.load_text()
        self.vocab = self.build_vocab()
        self.char2idx = self.build_char2idx()
        self.idx2char = self.build_idx2char(self.char2idx)
        self.text_as_int = self.text_to_int()
        self.ids_dataset = self.build_ids_dataset()
        self.seq_length = seq_length
        self.sequences = self.batch_sequence()
        self.dataset = None
    
    def get_text(self):
        return self.text
    
    def get_vocab(self):
        return self.vocab

    def get_char2idx(self):
        return self.char2idx
    
    def get_idx2char(self):
        return self.idx2char
    
    def get_text_as_int(self):
        return self.text_as_int
    
    def get_ids_dataset(self):
        return self.ids_dataset
    
    def get_sequences(self):
        return self.sequences
    
    def get_dataset(self):
        return self.dataset
    
    def set_dataset(self, dataset):
        self.dataset = dataset
        
    def load_text(self):
        text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        return text

    def build_vocab(self):
        return sorted(set(self.text))

    def build_char2idx(self):
        ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=self.vocab, mask_token=None)
        return ids_from_chars
    
    def build_idx2char(self, ids_from_chars):
        chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
            )
        return chars_from_ids

    def text_to_int(self):
        return self.char2idx(tf.strings.unicode_split(self.text, 'UTF-8'))
    
    def build_ids_dataset(self):
        all_ids = self.char2idx(tf.strings.unicode_split(self.text, 'UTF-8'))
        return tf.data.Dataset.from_tensor_slices(all_ids)
    
    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    
    def batch_sequence(self):
        sequences = self.ids_dataset.batch(self.seq_length+1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)
        return dataset
    
    def build_dataset(self, batch_size, buffer_size):
        self.dataset = (self.sequences
                        .shuffle(buffer_size)
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        )
        return self.dataset
    