from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain
import streamlit as st


from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

name_prompt=PromptTemplate(template="I want to open a restaurant for {cuisine} food. Please suggest me only one fancy name for this without explination.", input_variables=["cuisine"])
food_item_prompt = PromptTemplate(template="Based on the restaurant name '{restaurant_name}', suggest popular {cuisine} food items without description. Return them as comma-separated values.",  input_variables=["restaurant_name"])


name_chain = LLMChain(llm=llm, prompt=name_prompt,output_key="restaurant_name")
food_chain = LLMChain(llm=llm, prompt=food_item_prompt, output_key="menu_items")

chain = SequentialChain(chains=[name_chain, food_chain],
                        input_variables=['cuisine'],
                        output_variables=['restaurant_name','menu_items'])

# response = chain1.invoke({"cuisine":"indian"})

# print(response)
# print(response['restaurant_name'])
# print(response['menu_items'])

# streamlit app
st.title("Restaurant Name with Food Items Generator")
cuisine = st.sidebar.selectbox("pick a cuisine", ("Indian", "Italian", "Mexican", "Arabic", "American"))

def generate_restauran_name_and_items(cuisine):
    response = chain.invoke({"cuisine":cuisine})
    return response

if cuisine:
    response= generate_restauran_name_and_items(cuisine)
    st.header(response['restaurant_name'])
    manue= response['menu_items'].split(',')

    st.write("**Menue Items**")

    for item in manue:
        st.write("-", item)
    


