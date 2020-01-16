import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from GensimGloveVectorizer import GensimGloveVectorizer
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from nltk.tokenize import sent_tokenize, word_tokenize


class EncoderDecoderAttention:

    def __init__(self, embedding_size=300, encoder_output=300, decoder_output=None, max_output_sequence=17, glove_path=None, use_cnn=False, embedding_matrix=None, max_num_words=900):

        self.embedding_size =embedding_size
        self.encoder_output = encoder_output
        self.max_output_sequence = max_output_sequence

        if decoder_output is None:
            self.decoder_output = 2*self.encoder_output
        self.use_cnn = use_cnn
        self.max_num_words = max_num_words
        self.embedding_matrix = embedding_matrix
        self.vocab_size = None
        self.target_chars = ['d','t','fv','rs','l','r-ln','r-ny','t-psa','t-cons','[',']','alt-isda','alt-g','alt-s','prm','\n','0']
        self.num_decoder_chars = len(self.target_chars) - 1
        self.target_char_index = dict([(char, i) for i, char in enumerate(self.target_chars)])
        self.reverse_target_char_index = dict((i,char) for char, i in self.target_char_index.items())

        self.vectorizer = GensimGloveVectorizer(glove_path=glove_path)

        self.weight_path = None
        self.vectorizer_path = None
        self.model = None
        #self.model = self.get_model()

    def get_model(self):

        self.vocab_size = self.vectorizer.get_vocabulary_size()
        self.embedding_matrix = self.vectorizer.get_embedding_matrix()

        if self.use_cnn:
            self.repeator = RepeatVector(465)
        else:
            self.repeator = RepeatVector(self.max_num_words)

        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(self.max_output_sequence, activation="tanh")
        self.densor2 = Dense(1, activation="relu")
        self.activator = Activation(self.softmax, name="attention_weights")
        self.dotor = Dot(axes=1)

        encoder_input = Input(shape=(self.max_num_words,), name="encoder_input")

        mask = None
        if not self.use_cnn:
            shape = K.shape(encoder_input)
            mask = tf.ones((shape[0], self.max_num_words))
            mask = tf.where(tf.equal(encoder_input, 0.25), encoder_input, mask)

        embedding_layer = Embedding(self.vocab_size, self.embedding_size, mask_zero=(not self.use_cnn), trainable=False, weights=None if self.embedding_matrix is None else [self.embedding_matrix])(encoder_input)

        spatial_dropout = SpatialDropout1D(0.25)(embedding_layer)

        encoder_lstm_1 = None

        if self.use_cnn:
            cnn_layer = Conv1D(64, 6, padding="valid", activation="relu", strides=2)(spatial_dropout)
            encoder_lstm_1 = Bidirectional(LSTM(self.encoder_output, return_sequences=True, recurrent_dropout=0.10), name="encoder_lstm_1")(cnn_layer)
            encoder_lstm_2 = Bidirectional(LSTM(self.encoder_output, return_sequences=True), name="encoder_lstm_2")(encoder_lstm_1)
        else:
            encoder_lstm_1 = Bidirectional(LSTM(self.encoder_output, return_sequences=True, recurrent_dropout=0.10), name="encoder_lstm_1")(spatial_dropout)
            encoder_lstm_2 = Bidirectional(LSTM(self.encoder_output, return_sequences=True), name="encoder_lstm_2")(encoder_lstm_1)

        a = encoder_lstm_2

        decoder_lstm = LSTM(self.decoder_output, return_state=True, name="decoder_lstm")
        decoder_dense = Dense(self.num_decoder_chars, activation=self.softmax, name="output")

        outputs = []

        s0 = Input(shape=(self.decoder_output,), name="s0")
        c0 = Input(shape=(self.decoder_output,), name="c0")
        s = s0
        c = c0

        for t in range(self.max_output_sequence):

            context = self.one_step_attention(a, s, encoder_input, mask)

            s, _, c = decoder_lstm(context, initial_state=[s,c])

            s = Dropout(0.35)(s)
            out = decoder_dense(s)
            outputs.append(out)

        model = Model(inputs=[encoder_input,s0, c0], outputs=outputs)
        optimizer = Adadelta()


        model.compile(loss=self.custom_categorical_crossentropy, optimizer=optimizer, metrics=["categorical_crossentropy"])

        model.summary()

        return model

    def preprocess(self, X, y=None, go_character=None):

        X = pad_sequences(X, padding="post", truncating="post", maxlen=self.max_num_words)

        if y is not None:
            y_np = []
            for labels in y:
                row = [self.target_char_index[elem] for elem in labels]
                row.append(self.target_char_index["\n"])
                y_np.append(row)

            y = pad_sequences(y_np, value=self.target_char_index['0'], padding="post", truncating="post", maxlen=self.max_output_sequence)

            if go_character != None:
                y = np.insert(y, 0, self.target_char_index[go_character], axis=1)

            y = to_categorical(y)
            y = y[:, :, :-1]
            return X, y

        return X

    def fit(self, X, y, validation_data=None, validation_split=None, batch_size=44, epochs=1200):

        if self.vectorizer_path != None:
            self.vectorizer = GensimGloveVectorizer.load(self.vectorizer_path)

        X = self.vectorizer.fit(X)

        self.model = self.get_model()

        if self.weight_path != None:
            self.model.load_weights(self.weight_path)

        X, y = self.preprocess(X,y=y)

        print("X.shape:", X.shape)
        print("y.shape:", y.shape)

        m = X.shape[0]
        s0 = np.zeros((m, self.decoder_output))
        c0 = np.zeros((m, self.decoder_output))

        X_val = None
        y_val = None
        outputs_val = None
        m_val = None
        s0_val = None
        c0_val = None

        if validation_data is not None:
            print("Using validation data...")
            X_val, y_val = validation_data
            X_val, y_val = self.preprocess(X_val, y_val)

            m_val = X_val.shape[0]
            s0_val = np.zeros((m, self.decoder_output))
            c0_val = np.zeros((m, self.decoder_output))
            outputs_val = list(y_val.swapaxes(0,1))

        callback = CallbackActions(self.vectorizer)
        outputs = list(y.swapaxes(0,1))

        if validation_data is not None:
            self.model.fit([X,s0, c0], outputs, validation_data=([X_val, s0_val, c0_val]), callbacks=[callback], epochs=epochs, batch_size=batch_size, verbose=2)
        else:
            self.model.fit([X, s0, c0], outputs, validation_split=validation_split, callbacks=[callback], epochs=epochs, batch_size=batch_size, verbose=2)

        return

    def one_step_attention(self, a, s_prev, mask, use_rnn=True):

        s_prev = self.repeator(s_prev)
        concat = self.concatenator([s_prev, a])
        e = self.densor1(concat)
        energies = self.densor2(e)

        if not self.use_cnn:
            energies = Lambda(self.repeat_function, output_shape=(self.max_num_words,1), arguments={"mask":mask})([energies])

        alphas = self.activator(energies)
        context = self.dotor([alphas, a])

        return context

    def save(self, weight_path="data/weights/weights.h5", vectorizer_path="data/weights/vectorizer/"):
        self.model.save_weights(weight_path)
        self.vectorizer.save(vectorizer_path)
        return

    def load(self, weight_path=None, vectorizer_path=None):
        if weight_path is not None:
            self.weight_path = weight_path
            if self.model != None:
                self.model.load_weights(weight_path)
        if vectorizer_path is not None:
            self.vectorizer_path = vectorizer_path
            self.vectorizer = GensimGloveVectorizer.load(vectorizer_path)
        return

    def reset(self, load_weights=True):
        self.model = self.get_model()
        return

    def decode(self, y):
        y = np.argmax(y, axis=2)
        y = np.squeeze(y)

        labels_list = []
        for y_elem in y:
            labels = []
            for pred in y_elem:
                label = self.reverse_target_char_index[pred]
                labels.append(label)
            labels_list.append(labels)
        return labels_list

    def decode_label(self, num):
        return self.reverse_target_char_index[num]

    def is_not_bracket(self, label):
        return (label != "[" and label != "]")

    def is_bracket(self, label):
        return (label == "[" or label == "]")

    def get_label_string(self, y):
        if type(y[0]) is str:
            y = [y]

        labels = []

        for y_elem in y:
            text = ""
            i = 0
            prev_label = ""

            for y_label in y_elem:
                label = y_label

                if (self.is_not_bracket(label) and prev_label=="]") or (self.is_not_bracket(label) and self.is_not_bracket(prev_label)) or (label == "[" and self.is_not_bracket(prev_label)):
                    spnsp = " "
                elif label == "]" and self.is_not_bracket(prev_label) or (self.is_bracket(label) and self.is_bracket(prev_label)) or (self.is_not_bracket(label) and prev_label == "["):
                    spnsp = ""

                else:
                    spnsp = ""

                if label =="0" or label=="\n":
                    break

                text = text + spnsp + label

                prev_label = label
                i += 1

            labels.append(text)

            if len(y) == 1:
                return labels[0]

            return labels


    def predict(self, X, return_attention_map=False):
        X = self.preprocess(X)
        preds = self.predict_sequence(X)
        if return_attention_map:
            X_text = self.vectorizer.inverse_transform(X)
            attention_map = self.get_attention_map(X, X_text)

        if return_attention_map:
            return preds, attention_map
        else:
            return preds

        #return [np.array(prob) for prob in self.predict_proba(X)]

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict_sequence(self, X):
        m = X.shape[0]
        s0 = np.zeros((m, self.decoder_output))
        c0 = np.zeros((m, self.decoder_output))
        y_preds = self.model.predict([X, s0, c0], batch_size=800)
        y_preds = self.decode(y_preds)
        return y_preds

    def custom_categorical_crossentropy(self, y, y_pred):
        mask = K.equal(y, 0.)
        mask = 1.0 - K.cast(K.all(mask, axis=-1), K.floatx())
        seq_len = K.sum(mask, axis=-1)
        epsilon = K.epsilon()
        loss = categorical_crossentropy(y, y_pred)
        loss *= mask
        loss = K.sum(loss, axis=-1)
        loss /= K.cast(seq_len, K.floatx()) + epsilon
        return  loss

    def softmax(self, x, axis=1):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

    def get_attention_map(self, X, text):
        attention_maps = []
        predicted_labels = []
        for i in range(X.shape[0]):
            attention_map = np.zeros((self.max_output_sequence, self.max_num_words))
            ty, tx =  attention_map.shape
            s0 = np.zeros((1, self.decoder_output))
            c0 = np.zeros((1, self.decoder_output))
            attention_layer = self.model.get_layer("attention_weigts")

            X_single = np.expand_dims(X[i], axis=0)
            X_single = X_single.reshape((1,tx))

            f = K.function(self.model.inputs, [attention_layer.get_output_at(t) for t in range(ty)])
            r = f([X_single, s0, c0])

            for t in range(ty):
                for t_prime in range(tx):
                    attention_map[t][t_prime] = r[t][0,t_prime,0]

            prediction = self.model.predict([X_single, s0, c0])
            prediction = self.decode(prediction)

            print(prediction)

            text = [x for x in text[i]]

            labels = list(prediction)
            labels.append("")

            text.insert(0, "Labels")

            attention_df = pd.DataFrame(columns=[text])
            attention_df["Labels"] = labels

            for index, row in attention_df.iterrows():
                if index == len(attention_df.index) -1:
                    break
                attention_df.iloc[index,1:] = attention_map[index]
            attention_df.to_excel("data/attention_map_"+str(i))

            predicted_labels.append(prediction)
            attention_maps.append(attention_map)

        return attention_maps

    def repeat_function(self, x, mask=None):
        energies = x[0]
        padding = K.ones_like(energies)*(-2**32)
        mask = K.cast(K.tile(K.expand_dims(mask, axis=-1), [1,1,energies.shape[-1]]), "float32")
        return K.tf.where(K.tf.equal(mask,0.0), padding, energies)

def load_data(file_path, X_column="Text", y_column="Labels", dropna=True):
    data_df = pd.read_excel(file_path)
    if dropna:
        data_df = data_df.dropna(subset=[y_column])
        data_df = data_df.dropna(subset=[X_column])

    X = data_df[X_column].values
    y = data_df[y_column].values

    y = [ labels.replace("[", " [ ") \
                .replace("]", " ] ") \
                .replace("*", "") \
                .split() for labels in y ]

    return X, y, data_df

class CallbackActions(Callback):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        return

    def on_train_begin(self, logs={}):
        self.vectorizer.save("data/weights/vectorizer.vec")
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights("data/weights/weights.h5")
        loss = logs["loss"]
        if loss < 0.0150:
            self.model.save_weights("data/weights/weights_loss_"+"{0:.{1}f}".format(loss,4)+".h5")
        return

if __name__ == "__main__":
    X, y, data_df = load_data("data/RMBS.xlsx")
    attention_model = EncoderDecoderAttention() #glove_path="D:/Development/Embeddings/Glove/glove.840B.300d.txt")
    #attention_model.load("data/weights.h5")
    attention_model.fit(X,y)
    print()