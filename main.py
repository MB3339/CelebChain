## Integrate code with OpenAI API
# You need to have your own API key for open AI
import os 
from constants import openai_key
from langchain_community.llms import OpenAI

import streamlit as st
os.environ["OPENAI_API_KEY"]=openai_key

# Streamlit Frame work
st.title('Langchain Demo with OPENAI API')
input_text = st.text_input("Search the topic you want:")

## Open AI LLM Model
llm=OpenAI(temperature=0.8) # This is how much control agent should have while providing the response (range0-1)



if input_text:
   st.write(llm(input_text))

