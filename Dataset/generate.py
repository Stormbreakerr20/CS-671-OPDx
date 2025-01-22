import time
import json
from langchain.chat_models import ChatOpenAI
import pandas as pd
llm_name = "gpt-3.5-turbo"

import os
import openai
import sys

key = '' # your OpenAI API Key

llm = ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=key)
# llm.predict("Hello world!")

# open cases.json and read each case
# for each case, generate a conversation between a doctor and patient

# open data.csv in read mode and create a dataframe
with open('generated_data.csv', 'r') as f:
    df1 = pd.read_csv(f)


with open('cases.json', 'r') as f:
    cases = json.load(f)

# initialize empty dataframe
df2 = pd.DataFrame(columns=['case', 'conversation'])

for i in range(10000,10990):

    PROMPT = "You will have a conversation between a doctor (must be named Assistant) and patient (must be named Human), focusing on differential diagnosis based on the following patient history containing patient's symptoms followed by antecedents." + cases[str(i+1)] + "Assistant: Ask 1 question about main symptom details, covering nature, location, severity, and onset (use details verbatim from symptoms section) Ask 1 question about relevant medical history (use antecedents section) State the top 2 potential diagnoses from the given differential diagnosis list, along with their associated probabilities Ask 1 follow-up question related to a symptom detail that could help differentiate between the two potential diagnoses  Recommend 1 next step for diagnosis (e.g. test, examination, etc.)  Patient: Respond with complete statements from the symptoms and antecedents sections verbatim If asked about any missing details, do not provide any new information The conversation should strictly utilize the information provided in the patient history sections. No new clinical details should be introduced. The dialog must end with the Assistant ' The dialogue must be in the following format: ###Assistant: What are your symptoms? ### Human: I have chest pain even at rest. The pain is a knife stroke in my upper chest, breast (R) and breast (L). On a scale of 0-10, the pain intensity is 7 and the pace it appeared is 9.### Assistant: Any relevant medical history?### Human: I have had a spontaneous pneumothorax. I smoke cigarettes. I have a chronic obstructive pulmonary disease. Some family members have had a pneumothorax.### Assistant: Based on your history, the two most likely diagnoses are unstable angina at 26.2% probability, and spontaneous pneumothorax at 6% probability. Does the pain increase or decrease with physical exertion?### Human: I have symptoms that increase with physical exertion but alleviate with rest.### Assistant: Let's get a chest X-ray to examine for a potential pneumothorax."

    # time.sleep(1)
    response = llm.predict(PROMPT)
    conv = str(response)
    # print(conv)
    
    # add case and conversation to dataframe
    df2.loc[i] = [i+1, conv]
    print(i)

# append df to df1
appended = pd.concat([df1, df2], axis=0)

# write dataframe to generated_data.csv
appended.to_csv('generated_data.csv', index=False)