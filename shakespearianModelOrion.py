import tensorflow as tf
import numpy as np
import os
import time

class Orion(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

class singleStep(tf.keras.Model):
    def __init__(self, model, ids2chars, chars2idx, temp=1.0):
        super().__init__()
        self.temp = temp
        self.model = model
        self.ids2chars = ids2chars
        self.chars2idx = chars2idx

        # Create a mask to prevent undesired chars in the output
        skip_ids = self.chars2idx(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
        # Match the shape to the vocabulary
            dense_shape=[len(chars2idx.get_vocabulary())])
        self.mask = tf.sparse.to_dense(sparse_mask)


    @tf.function
    def gen_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.chars2idx(input_chars).to_tensor()

        # Run the model.
        log_pred, states = self.model(inputs=input_ids, states=states, return_state=True)

        #USe lastest pred
        log_pred = log_pred[:, -1, :]
        log_pred = log_pred/self.temp
        log_pred = log_pred + self.mask
        #Use the mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(log_pred, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states
