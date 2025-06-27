from langchain.chains import LLMChain, SequentialChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from constants import openai_key
import streamlit as st

# Initialize LLM
llm = OpenAI(openai_api_key=openai_key, temperature=0.8)# temp is showing how muc controll agent has (value range 0-1)

# Prompt 1: Info about person
prompt1 = PromptTemplate(
    input_variables=["name"],
    template="Tell me about {name}."
)
person_memory = ConversationBufferMemory(input_key='name',memory_key="chat_history", return_messages=True)

chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="person", memory=person_memory) 



# Prompt 2: DOB of person — still uses `name`
prompt2 = PromptTemplate(
    input_variables=["person"],
    template="What is the {person} birthdate?"
)
dob_memory = ConversationBufferMemory(input_key='person',memory_key="chat_history", return_messages=True)

chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="dob", memory=dob_memory)

# Prompt 3: Events around that dob
prompt3 = PromptTemplate(
    input_variables=["dob"],
    template="What major world events happened around {dob}?"
)
event_memory = ConversationBufferMemory(input_key='dob',memory_key="description_history", return_messages=True)

chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="events",memory=event_memory)

# ✅ Combine chains with `return_all=True`
sequential_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["name"],
    output_variables=["person", "dob", "events"],
    verbose=True,
    return_all=True 
)

# Streamlit UI
st.title("Celebrity Info Lookup")
name_input = st.text_input("Enter a celebrity's name:")

if name_input:
    try:
        result = sequential_chain.invoke({"name": name_input})  
        st.subheader(" Info")
        st.write(result["person"])
        
        st.subheader(" Date of Birth")
        st.write(result["dob"])
        
        st.subheader(" Events Around That Time")
        st.write(result["events"])
    except Exception as e:
        st.error(f" Error: {e}")



