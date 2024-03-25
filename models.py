import tensorflow as tf

from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import regularizers

import pandas as pd



EMBEDDING_DIM = 128 

purchase_history = pd.read_csv('model-training/purchase_history.csv')
customer_interactions = pd.read_csv('model-training/customer_interactions.csv')
product_details = pd.read_csv('model-training/product_details.csv')

data = purchase_history.merge(customer_interactions)
data = data.merge(product_details)

n_products = data.product_id.nunique()
n_users = data.user_id.nunique()
n_category =  data.category_id.nunique()
unique_products = list(data.product_id.unique())
unique_users = list(data.user_id.unique())
unique_category = list(data.category_id.unique())

class Config:
    desc = "with session context + category embedding + class weights"
    embedding= 16
    dropout= 0.1
    num_of_deep_layers= 4
    l2= 0
config = Config()



def get_model():

    EMBEDDING_DIM = config.embedding

    # WIDE MODEL
    wide_model = Sequential([Dense(EMBEDDING_DIM * 2, use_bias='false', 
                            kernel_regularizer=regularizers.l2(config.l2)),
                            Dropout(config.dropout)])

    # DEEP MODEL
    deep_layers = []
    for i in range(config.num_of_deep_layers):
        deep_layers += [Dense(EMBEDDING_DIM, activation='relu', use_bias='false', kernel_regularizer=regularizers.l2(config.l2)),
                        Dropout(config.dropout)]
                
    deep_model = Sequential(deep_layers)

    # input layers
    product_input = Input(shape=[1], name='product_id')
    user_input = Input(shape=[1], name='user_id')

    page_views_input = Input(shape=[1], name='page_views')
    session_time_input = Input(shape=[1], name='session_time')

    price_input = Input(shape=[1], name='price')

    category_input = Input(shape=[1], name='category_id')
    category_input_int = tf.keras.layers.IntegerLookup(vocabulary=unique_category, num_oov_indices=1, mask_token=None)(category_input)
    category_embedding = Embedding(n_products+1, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(config.l2))(category_input_int)

    session_context = concatenate([price_input, page_views_input, session_time_input])
    session_context = tf.keras.layers.BatchNormalization()(session_context)
    session_context = tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu')(session_context)

    product_input_int = tf.keras.layers.IntegerLookup(vocabulary=unique_products, num_oov_indices=1, mask_token=None)(product_input)
    user_input_int = tf.keras.layers.IntegerLookup(vocabulary=unique_users, num_oov_indices=1, mask_token=None)(user_input)

    # # Convert strings to integers
    # product_input_int = product_lookup(product_input)
    # user_input_int = user_lookup(user_input)

    # embedding layers 
    product_embedding = Embedding(n_products+1, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(config.l2))(product_input_int)
    user_embedding = Embedding(n_users+1, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(config.l2))(user_input_int)

    # flatten the embeddings
    product_flat = Flatten()(product_embedding)
    user_flat = Flatten()(user_embedding)
    category_embedding = Flatten()(category_embedding)

    # wide and deep model
    concat = concatenate([product_flat, user_flat, category_embedding])
    wide_output = wide_model(concat)
    deep_output = deep_model(concat)

    # output layer
    concat_2 = concatenate([wide_output, deep_output])
    embeddings_output = tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu',  kernel_regularizer=regularizers.l2(config.l2), use_bias='false')(concat_2)
    concat_3 = concatenate([embeddings_output, session_context])
    output = Dense(1, activation='sigmoid', use_bias='false')(concat_3)
    #output = sigmoid(Dot(1)([wide_output, deep_output]))

    # the model
    model = Model([product_input, user_input, page_views_input, session_time_input, price_input, category_input], [output])
    return model