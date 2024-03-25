import tensorflow as tf 
import pandas as pd 
import numpy as np

import models

model = models.get_model()
model.load_weights('model-training/final_model.h5')

product_details = pd.read_csv('model-training/product_details.csv')
customer_interactions = pd.read_csv('model-training/customer_interactions.csv')
user_details = customer_interactions.loc[customer_interactions.user_id.drop_duplicates().index]

def get_product_recommendation(user_id, page_views, session_time, products=product_details, model=model):
    """
    Find which products a given user might are likely to buy

    """
    products = products.copy()
    user_ids = [user_id] * len(products)
    page_views = [page_views] * len(products)
    session_time = [session_time] * len(products)

    input_ = pd.DataFrame({'product_id': list(products.product_id), 
                           'user_id': user_ids,
                           'page_views': page_views,
                           'session_time': session_time,
                           'price': list(products.price),
                           'category_id': list(products.category_id)})
    
    results = model([input_['product_id'], 
                     input_['user_id'],
                     input_['page_views'].values.reshape(-1, 1),
                     input_['session_time'].values.reshape(-1, 1),
                     input_['price'].values.reshape(-1, 1),
                     input_['category_id']
                    ]).numpy().reshape(-1)

    products['purchase_proba'] = pd.Series(results, index=products.index)
    products = products.sort_values('purchase_proba', ascending=False)
    
    return products



def get_user_recommendation(product_id, price, category_id, users=user_details, model=model):
    """
    Find which users are likely to buy which product
    
    """
    users = users.copy()
    product_ids = [product_id] * len(users)
    prices = [price] * len(users)
    category_ids = [category_id] * len(users)

    
    input_ = pd.DataFrame({'product_id': product_ids, 
                           'user_id': list(users.user_id),
                           'page_views': list(users.page_views),
                           'session_time': list(users.session_time),
                           'price': prices,
                           'category_id': category_ids})
    
    results = model([input_['product_id'], 
                     input_['user_id'],
                     input_['page_views'].values.reshape(-1, 1),
                     input_['session_time'].values.reshape(-1, 1),
                     input_['price'].values.reshape(-1, 1),
                     input_['category_id']
                    ]).numpy().reshape(-1)
    
    users['purchase_proba'] = pd.Series(results, index=users.index)
    users = users.sort_values('purchase_proba', ascending=False)
    
    return users