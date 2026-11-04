"""Arquitectura seq2seq (encoder-decoder) parametrizable: RNN simple o LSTM.

Mismo esqueleto para ambos modelos (embedding -> capa recurrente -> capa de
salida) para que la comparacion RNN simple vs LSTM sea justa: cambia solo el
tipo de celda recurrente, mantenemos iguales dimension de embedding, unidades
y datos de entrenamiento.
"""
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.models import Model

EMBEDDING_DIM = 128
UNITS = 256


def _recurrent_layer(cell_type, units, return_state, return_sequences=True, name=None):
    if cell_type == "rnn":
        return SimpleRNN(units, return_sequences=return_sequences, return_state=return_state, name=name)
    if cell_type == "lstm":
        return LSTM(units, return_sequences=return_sequences, return_state=return_state, name=name)
    raise ValueError(f"cell_type desconocido: {cell_type}")


def build_training_model(vocab_size_es, vocab_size_qu, cell_type, embedding_dim=EMBEDDING_DIM, units=UNITS):
    """Construye el modelo de entrenamiento con teacher forcing."""
    # --- Encoder (espanol) ---
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    encoder_emb = Embedding(vocab_size_es, embedding_dim, mask_zero=True, name="encoder_embedding")(encoder_inputs)

    if cell_type == "lstm":
        encoder_outputs, state_h, state_c = _recurrent_layer(cell_type, units, return_state=True, name="encoder_lstm")(encoder_emb)
        encoder_states = [state_h, state_c]
    else:
        encoder_outputs, state_h = _recurrent_layer(cell_type, units, return_state=True, name="encoder_rnn")(encoder_emb)
        encoder_states = [state_h]

    # --- Decoder (quechua) ---
    decoder_inputs = Input(shape=(None,), name="decoder_input")
    decoder_embedding_layer = Embedding(vocab_size_qu, embedding_dim, mask_zero=True, name="decoder_embedding")
    decoder_emb = decoder_embedding_layer(decoder_inputs)

    decoder_rnn_layer = _recurrent_layer(cell_type, units, return_state=True, return_sequences=True, name=f"decoder_{cell_type}")
    if cell_type == "lstm":
        decoder_outputs, _, _ = decoder_rnn_layer(decoder_emb, initial_state=encoder_states)
    else:
        decoder_outputs, _ = decoder_rnn_layer(decoder_emb, initial_state=encoder_states)

    decoder_dense = Dense(vocab_size_qu, activation="softmax", name="decoder_output")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name=f"seq2seq_{cell_type}")

    # --- Modelos de inferencia (decodificacion paso a paso) ---
    encoder_model = Model(encoder_inputs, encoder_states, name=f"encoder_{cell_type}")

    if cell_type == "lstm":
        decoder_state_input_h = Input(shape=(units,))
        decoder_state_input_c = Input(shape=(units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        dec_emb2 = decoder_embedding_layer(decoder_inputs)
        dec_out2, state_h2, state_c2 = decoder_rnn_layer(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
    else:
        decoder_state_input_h = Input(shape=(units,))
        decoder_states_inputs = [decoder_state_input_h]
        dec_emb2 = decoder_embedding_layer(decoder_inputs)
        dec_out2, state_h2 = decoder_rnn_layer(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2]

    dec_out2 = decoder_dense(dec_out2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [dec_out2] + decoder_states2, name=f"decoder_{cell_type}")

    return model, encoder_model, decoder_model
