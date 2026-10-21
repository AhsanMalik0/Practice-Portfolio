import numpy as np
import tensorflow as tf
import math
from tensorflow import keras
import numpy as np


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, maxlen, dmodel):
        super(PositionalEncoding, self).__init__()
        #Dimension of the vector
        self.dmodel = dmodel
        # Positions, is the columns vectore that represent the position of the token in sequence from "zero" to "maxlen"
        # Position vevtor hold the position of each token in sequence (Position indes of each token)
        positions = np.arange(maxlen)[:, np.newaxis]
        # index are row vector represent the dimension indices of positional encoding
        index = np.arange(dmodel)[np.newaxis, :]
        #angles Rates, The scaling factors applied to each dimension based on the positional index. 
        # These scale the position values differently for each dimension
        angle_rates = 1/np.power(10000, (2 * (index // 2)) / np.float32(dmodel))
        # Angle_rads, is scaling factors that applied to each dimension based on the positional index. 
        angle_rads = positions*angle_rates
        # Apply sign on even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos on odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        # ready the positional embiddibg for input and change the type to tensorflow f.32
        self.positional_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]


class ScaleDotProductAttention(keras.layers.Layer):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        
    def call(self, Query, Key, Value, Mask=None):
        matmul_QK = tf.matmul(Query,Key, transpose_b=True)
        Depth = tf.cast(tf.shape(Key)[-1], tf.float32)
        logits = matmul_QK / tf.math.sqrt(Depth)
        
        if Mask is not None:
            logits += (Mask * -1e9)
        Attention_weights = tf.nn.softmax(logits, axis=-1)
        Output = tf.matmul(Attention_weights,Value)
        return Output, Attention_weights



class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_heads, dmodel):
        super(MultiHeadAttention, self).__init__()
        self.dmodel = dmodel
        self.num_heads = num_heads
        #
        assert dmodel % self.num_heads == 0
        self.Depth = dmodel // self.num_heads
        #
        self.Query_Den = keras.layers.Dense(units=dmodel)
        self.Key_Den = keras.layers.Dense(units=dmodel)
        self.Val_Den = keras.layers.Dense(units=dmodel)
        #
        self.attention = ScaleDotProductAttention()
        #
        self.output_Den = keras.layers.Dense(units=dmodel)
        #
    def SplitHeads(self, X, Batch_Size):
        x = tf.reshape(X, shape=(Batch_Size, -1,self.num_heads, self.Depth))
        return tf.transpose(x, perm=[0,2,1,3])
        
    def call(self, Query, Key, Value, Mask=None):
        batch_size = tf.shape(Query)[0]
        # Dense layer on all of inputs
        query = self.Query_Den(Query)
        key = self.Key_Den(Key)
        value = self.Val_Den(Value)
        # splits "Query", "Key" and "Value" into multiple heads
        query = self.SplitHeads(query, batch_size)
        key = self.SplitHeads(key, batch_size)
        value = self.SplitHeads(value,batch_size)
        #
        Output, Weights = self.attention(query,key,value,Mask)
        #
        Output = tf.transpose(Output,perm=[0,2,1,3])
        Output = tf.reshape(Output,(batch_size, -1, self.dmodel))
        #
        Output = self.output_Den(Output)
        #
        return Output, Weights


class AddNorm(keras.layers.Layer):
    def __init__(self, Epsilon=1e-6):
        super(AddNorm,self).__init__()
        self.layer_norm = keras.layers.LayerNormalization(epsilon=Epsilon)
    def call(self, Attn_out, Pos_out):
        return self.layer_norm(Attn_out+Pos_out)

class FeedForwardNetwork(keras.layers.Layer):
    def __init__(self, dmodel, dff):
        super(FeedForwardNetwork, self).__init__()
        self.dff = dff
        #
        self.Dense1 = keras.layers.Dense(units=self.dff, activation='relu')
        #self.Dense2 = keras.layers.Dense(units=self.dff//2, activation='relu')
        self.out = keras.layers.Dense(units=dmodel)
    def call(self, norm_out):
        return self.out(self.Dense1(norm_out))

class EncoderLayer(keras.layers.Layer):
    def __init__(self, dmodel, dff, num_heads, dropout_rate = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.MHA = MultiHeadAttention(num_heads=num_heads, dmodel=dmodel)
        self.Norm1 = AddNorm()
        self.Norm2 = AddNorm()
        self.feedforward = FeedForwardNetwork(dmodel=dmodel, dff=dff)
        self.Dropout1 = keras.layers.Dropout(dropout_rate)
        self.Dropout2 = keras.layers.Dropout(dropout_rate)
        #
    def call(self, Pos_Out, Mask=None):
        #
        Attn_output, _ = self.MHA(Pos_Out, Pos_Out, Pos_Out, Mask)   # In Attention, our Input are feeded as three inputs 
        #                                                      #            
        Attn_output = self.Dropout1(Attn_output)               #            Dropout apply on Attention output weights     
        #                                                      #
        AddNorm_out_1 = self.Norm1(Attn_output,Pos_Out)        #            Feed the input and attention output in Add & Norm Layer
        #                                                      #
        feedfwd = self.feedforward(AddNorm_out_1)              #            Feed the Add & Norm output in FeedForward Network
        #                                                      #
        fedfwd = self.Dropout2(feedfwd)                        #            Apply Dropout on FeedForward Weights
        #                                                      #
        AddNorm_out_2 = self.Norm2(feedfwd, AddNorm_out_1)     #       feed the feedforward and Add & Norm output again in Add & Norm Layer
        
        return AddNorm_out_2



class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, dmodel, dff, num_heads, maxlen, dropout_rate=0.1):
        super(Encoder,self).__init__()
        self.num_layers = num_layers
        self.dmodel = dmodel
        self.maxlen = maxlen
        #
        self.encoder = [EncoderLayer(dff=dff,dmodel=dmodel,num_heads=num_heads, dropout_rate=0.1) for _ in range(num_layers)]
        self.Positional_Encoding = PositionalEncoding(maxlen=maxlen, dmodel=dmodel)
    def call(self, X, Mask=None):
        for i in range(self.num_layers):
            X = self.encoder[i](X,Mask)
        return X




class DecoderLayer(keras.layers.Layer):
    def __init__(self, dmodel, dff, num_heads, dropout_rate = 0.1):
        super(DecoderLayer,self).__init__()

        self.MHA1 = MultiHeadAttention(num_heads=num_heads, dmodel=dmodel)
        self.MHA2 = MultiHeadAttention(num_heads=num_heads, dmodel=dmodel)
        
        self.Norm1 = AddNorm()
        self.Norm2 = AddNorm()
        self.Norm3 = AddNorm()
        #
        self.feedforward = FeedForwardNetwork(dmodel=dmodel, dff=dff)
        #
        self.Dropout1 = keras.layers.Dropout(dropout_rate)
        self.Dropout2 = keras.layers.Dropout(dropout_rate)
        self.Dropout3 = keras.layers.Dropout(dropout_rate)
        
    def call(self, decoder_input, encoder_output, look_ahead_mask=None, padding_mask=None):
        #First block in Decoder
        Attn_output1, _ = self.MHA1(decoder_input, decoder_input, decoder_input, look_ahead_mask)
        Attn_output1 = self.Dropout1(Attn_output1)
        AddNorm_out_1 = self.Norm1(Attn_output1,decoder_input)
        #
        Attn_output2, _ = self.MHA2(AddNorm_out_1, encoder_output, encoder_output, padding_mask)
        Attn_output2 = self.Dropout1(Attn_output2)
        AddNorm_out_2 = self.Norm2(Attn_output2,AddNorm_out_1)
        #
        feedfrwd = self.feedforward(AddNorm_out_2)
        feedfrwd = self.Dropout1(feedfrwd)
        AddNorm_out_3 = self.Norm3(feedfrwd, AddNorm_out_2)
        return AddNorm_out_3


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, dmodel, dff, num_heads, maxlen, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.dmodel = dmodel
        self.maxlen = maxlen
        #self.num_layers = num_layers
        #self.num_heads = num_heads
        #self.maxlen = maxlen
        #
        self.decoder = [DecoderLayer(dff=dff, dmodel=dmodel,num_heads=num_heads, dropout_rate=0.1) for _ in range(num_layers)]
    #
    def call(self, inputs, encoder_outputs, look_ahead_mask=None, padding_mask=None):
        dec_input = inputs
        for i in range(self.num_layers):
            dec_input = self.decoder[i](decoder_input=dec_input, encoder_output=encoder_outputs,  look_ahead_mask=None, padding_mask=None)
        return dec_input


class Transformer_Out(keras.layers.Layer):
    def __init__(self, vocab_size):
        super(Transformer_Out, self).__init__()
        self.Dens = keras.layers.Dense(units=vocab_size)
    def call(self, Decoder_output):
        return self.Dens(Decoder_output)

def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # shape: (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # shape: (seq_len, seq_len)

def create_combined_mask(seq):
    look_ahead = create_look_ahead_mask(tf.shape(seq)[1])
    padding = create_padding_mask(seq)
    return tf.maximum(look_ahead, padding)  # shape: (batch_size, 1, seq_len, seq_len)

class Transformer(keras.Model):
    def __init__(self, num_layers, dmodel, dff, num_heads, vocab_size, maxlen, masks=False, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.masks = masks
        self.Embedding = keras.layers.Embedding(vocab_size, dmodel)
        self.Positional_Encoding = PositionalEncoding(maxlen=maxlen, dmodel=dmodel)
        self.Encoder_ = Encoder(num_layers=num_layers, dmodel=dmodel, dff=dff,
                                maxlen=maxlen, num_heads=num_heads, dropout_rate=dropout_rate)
        self.Decoder_ = Decoder(num_layers=num_layers, dmodel=dmodel, dff=dff,
                                num_heads=num_heads, maxlen=maxlen, dropout_rate=dropout_rate)
        self.Out_Put = Transformer_Out(vocab_size=vocab_size)

    def call(self, inputs, training=False):
        enc_inputs = inputs['inputs']
        dec_inputs = inputs['dec_inputs']
        # Create masks
        if self.masks == True:
            enc_padding_mask = create_padding_mask(enc_inputs)
            look_ahead_mask = create_look_ahead_mask(tf.shape(dec_inputs)[1])
            dec_target_padding_mask = create_padding_mask(dec_inputs)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        else:
            enc_padding_mask = None
            look_ahead_mask = None
            combined_mask = None

        enc_embed = self.Positional_Encoding(self.Embedding(enc_inputs))
        dec_embed = self.Positional_Encoding(self.Embedding(dec_inputs))
        
        # Encoder and Decoder
        enc_output = self.Encoder_(enc_embed, Mask=enc_padding_mask)
        dec_output = self.Decoder_(dec_embed, encoder_outputs=enc_output,
                                   look_ahead_mask=combined_mask,
                                   padding_mask=enc_padding_mask)
        
        return self.Out_Put(dec_output)


if __name__ == "__main__":
    # Define model hyperparameters
    num_layers = 96
    d_model = 128
    dff = 512
    num_heads = 4
    vocab_size = 10000  # Adjust based on your dataset
    max_len = 50  # Maximum sequence length

    # Initialize Transformer Model
    transformer = Transformer(
        num_layers=num_layers,
        dmodel=d_model,
        dff=dff,
        num_heads=num_heads,
        vocab_size=vocab_size,
        maxlen=max_len
    )


