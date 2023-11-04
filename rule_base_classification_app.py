import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px

st.set_page_config(
    page_title="Ruled Based Classification",
    page_icon="👩‍💻",
    layout="centered",
    initial_sidebar_state="expanded",
)

image = Image.open("C:/Users/yasmi/PycharmProjects/pythonProject13/analysis.png")
st.image(image, width=200)



markdown_text = f"""
<h1 style="font-size: 36px;">Calculating Potential Customer Return with Rule-Based Classification</h1>
<h4 style="font-size: 18px;">Business Problem</h4>
<p style="font-size: 16px;">A gaming company wishes to create level-based new customer personas using certain characteristics of its customers and to create segments based on these new customer personas. The company aims to estimate how much the potential new customers, based on these segments, can potentially generate in terms of revenue</p>
<p style="font-size: 16px;">For example, it is desired to determine how much a 25-year-old male user from Turkey who uses IOS can potentially generate on average.</p>
<h4 style="font-size: 18px;">Dataset Story</h4>
<p style="font-size: 16px;">Persona.csv data set contains the prices of the products sold by an international game company and some demographic information of the users who purchased these products. The data set consists of records created in each sales transaction. This means the table is not deduplicated. In other words, a user with certain demographic characteristics may have made more than one purchase.</p>
"""

st.markdown(markdown_text, unsafe_allow_html=True)

st.markdown("""
- Price: Customer's spending amount
- Source: The type of device the customer is connected to
- Sex: Customer's gender
- Country: Customer's country
- Age: Customer's age
""")

df = pd.read_csv('C:/Users/yasmi/Desktop/kural-tabanli-siniflandirma\persona.csv')

def check_dataframe(df, head=5):
    st.subheader("Shape of Dataset")
    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])
    st.subheader("Types of Columns")
    st.write(df.dtypes)
    st.subheader("First 5 Rows")
    st.write(df.head(head))
    st.subheader("Last 5 Rows")
    st.write(df.tail(head))
    st.subheader("Number of Null Values In The Dataset")
    st.write(pd.DataFrame(df.isnull().sum()))
    st.subheader("Summary Statistics")
    st.write(df.describe().T)


st.header('General Information of Dataset')
check_dataframe(df)


st.header("Categorical Variable Analysis")
st.subheader("SOURCE")
fig = px.histogram(df, x="SOURCE", color="SOURCE", nbins=20)
st.plotly_chart(fig)
st.subheader("COUNTRY")
fig = px.histogram(df, x="COUNTRY", color="COUNTRY", nbins=20)
st.plotly_chart(fig)
st.subheader("Sex")
fig = px.histogram(df, x="SEX", color="SEX", nbins=20)
st.plotly_chart(fig)

st.header("Numeric Variable Analysis")
st.subheader("Age Distribution")
fig = px.histogram(df, x="AGE", nbins=20)
st.plotly_chart(fig)
st.subheader("Age and Price Distribution")
fig = px.scatter(df, x="AGE", y="PRICE", color="SEX")
st.plotly_chart(fig)



col1, col2 = st.columns(2)

with col1:
    agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", axis=0,
                                                                                                   ascending=False)
    agg_df = agg_df.reset_index()
    st.text('Finding the average price by Country, Gender, Source, and Age')
    st.write(agg_df)


with col2:
    bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
    mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
    agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
    st.text('Converting age variable to categorical variable')
    st.write(agg_df)


agg_df["CUSTOMER_LEVEL_BASED"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df = agg_df[["CUSTOMER_LEVEL_BASED", "PRICE"]]
st.text('Creating a customer_level_based column and finding the average price accordingly')
st.write(agg_df)


agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})
st.text('Dividing the dataset into four separate segments based on the price column and calculating the mean value within each group, considering the price variable')
st.write(agg_df)

st.sidebar.title("New Customer Information")
country = st.sidebar.selectbox("Select Country", sorted(df['COUNTRY'].str.upper().unique()))
source = st.sidebar.selectbox("Select Source (OS)", sorted(df['SOURCE'].str.upper().unique()))
sex = st.sidebar.selectbox("Select Gender", sorted(df['SEX'].str.upper().unique()))
age = st.sidebar.number_input("Enter Age", min_value=0, max_value=100, value=18)

if st.sidebar.button("Save"):
    # Save the selected filters to a dictionary
    filters = {"Country": country, "Source (OS)": source, "Gender": sex, "Age": age}
    # Print the filters and a success message
    st.sidebar.write("New user added!:", filters)
    st.sidebar.success("Data insertion saved successfully!")

    new_user_df = [[country, source, sex, age]]
    new_user_df = pd.DataFrame(new_user_df, columns=["COUNTRY", "SOURCE", "SEX", "AGE"])

    new_user_df["AGE_CAT"] = pd.cut(new_user_df["AGE"], [0, 18, 23, 30, 40, 70],
                                    labels=['0_18', '19_23', '24_30', '31_40', '41_70'])
    age_cat = new_user_df["AGE_CAT"][0]

    new_user = (country + "_" + source + "_" + sex + "_" + age_cat).upper()
    price = agg_df[agg_df["CUSTOMER_LEVEL_BASED"] == new_user].reset_index(drop=True)


    def new_customer(dataframe, new_user):
        st.subheader('Segment and price prediction:')
        st.info(f'New User Information: {new_user}')
        if price.empty:
            st.error("No matching user found in the dataset.")
        else:
            st.success("Mean Price for New Customer: " + str(format(price["PRICE"][0], ".2f")) + "$")
            st.success("Segment for New Customer: " + str(price["SEGMENT"][0]))


    new_customer(agg_df, new_user)