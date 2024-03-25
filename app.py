import time
import streamlit as st

import pandas as pd
from pipeline import get_product_recommendation, get_user_recommendation



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


def recommendation_system():
    st.title("TerraStore - Recommendation System Demo")

    user_id = st.selectbox(
        "Which user id would you want to see?",
        unique_users,
        index=None,
        placeholder="Select user id",
    )

    # Session details
    page_views = st.number_input('How many pages has the user visited in this session?', min_value=0, step=1)

    session_time = st.number_input('How long has the user session last?', min_value=0, step=1)

    if user_id is not None:
        recommendations = get_product_recommendation(
            user_id=user_id, 
            page_views=page_views, 
            session_time=session_time
        )
        recommendations['category_id'] = recommendations['category_id'].astype(str)

        st.write("#### Top 10 products for user {}".format(user_id))
        st.dataframe(
            recommendations.head(10),
            hide_index=True,
            column_config={
            "product_id": st.column_config.NumberColumn(
                "Product ID",
                format="%d",
            )
        })

        st.write("#### Top 10 categories for user {}".format(user_id))
        st.dataframe(
            pd.Series(recommendations.category_id.unique()[:10], name='Category ID'),
            hide_index=True)
        

def cashback_and_discount_recommendation():
    st.title("TerraStore - Which users deserve discounts?")

    st.write("This page helps you to find which users are more likely to buy a given products. You could also tune the prices of the products to see how users might react.")

    product_id = st.selectbox(
        "Which product id would you want to see?",
        unique_products,
        index=None,
        placeholder="Select product id",
    )

    # Session details
    if product_id:
        price_value = product_details[product_details.product_id==product_id].price.iloc[0]
        category_id_value = product_details[product_details.product_id==product_id].category_id.iloc[0]
        category_id_index = unique_category.index(category_id_value)
    else: 
        price_value = None
        category_id_index = None
    price = st.number_input('How much does it cost?', value=price_value, min_value=0.0)

    category_id = st.selectbox(
        "In which category does this product belong?",
        unique_category,
        index=category_id_index,
        placeholder="Select category id",
    )
    

    if product_id is not None:
        recommendations = get_user_recommendation(
            product_id=product_id, 
            price=price, 
            category_id=int(category_id)
        )
        # recommendations['category_id'] = recommendations['category_id'].astype(str)

        st.write("#### Top 20 users who might buy {}".format(product_id))
        st.dataframe(
            recommendations.head(20),
            hide_index=True,
            column_config={
            "user_id": st.column_config.NumberColumn(
                "User ID",
                format="%d",
            )
        })


def main():
    page_names_to_funcs = {
        # "From Uploaded Picture": from_picture,
        "Recommendation system": recommendation_system,
        "Cashback & Discount": cashback_and_discount_recommendation
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Recommendation  System - Technical Interview", page_icon=":pencil2:"
    )
    # st.sidebar.subheader("Configuration")
    main()