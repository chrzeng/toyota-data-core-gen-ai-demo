#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install matplotlib


# In[1]:


import vertexai
from vertexai.preview.language_models import TextGenerationModel, ChatModel, CodeGenerationModel
import pandas as pd
from google.cloud import bigquery
import db_dtypes
import base64
from pathlib import Path
import matplotlib.pyplot as plt
from langchain.vectorstores import Chroma
import uuid  
import os
import time
from retry import retry
import pandas as pd
import numpy as np
from ast import literal_eval
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate


# In[63]:


#!pip install plotly.express
#!pip install plotly.graph_objs 


# In[85]:


import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
import re
from PIL import Image
import random
from retry import retry
import plotly.express as px
#import plotly.graph_objs as go


# In[106]:


timestamp = time.strftime("%Y%m%d%H%M%S")
filename = f"image_{timestamp}.png"


# In[86]:


def main_fun(ques):

    b = ques

    @retry(tries=2)
    def predict_large_language_model_sample( project_id: str, model_name: str, temperature: float, max_decode_steps: int, top_p: float, top_k: int, content: str, location: str = "us-central1", tuned_model_name: str = "",) :

        """Predict using a Large Language Model."""

        vertexai.init(project=project_id, location=location)

        model = TextGenerationModel.from_pretrained(model_name)

        if tuned_model_name:

          model = model.get_tuned_model(tuned_model_name)

        response = model.predict(content, temperature=temperature, max_output_tokens=max_decode_steps, top_k=top_k, top_p=top_p,)

        #print(f" {response.text}")

        output = response.text
       

        output = output.replace('sql','')
        

        output = output.replace('```','')
        

        return output

        

    data = predict_large_language_model_sample("us-gcp-ame-con-c2aaa-npd-1", "text-bison@001", 0, 962, 0.95, 40, f'''
    
    your_vision: " You are a chatbot for toyota vehicle hold inquary"
    your_mission: " your mission is to provide executable queries from the question {b} "
        
    return only the executable sql query for this question, {b} in bigquery? 
    
    CREATE TABLE ds.inventory (
        Locatoin_ID STRING,
        Part_ID STRING,
        Location STRING,
        Part_Number STRING,
        Part_Description STRING,
        Part_Age STRING,
        Location_Type STRING,
        Forecast_Period_ID DATE,
        History_ID DATE,
        Primary_Volume_Type STRING,
        Volume_Type STRING,
        Model_Name STRING,
        Forecast_Quantity FLOAT64,
        Forecast_Period_Type STRING,
        Inventory_on_hand FLOAT64,
        Error_Description STRING,
        Segmentation_ID STRING,
        Segmentation_Type STRING,
        Cost FLOAT64,
        Auto_SP_AP STRING,
        Min_Safety_Stock_Days FLOAT64,
        Min_Safety_Stock_Qty FLOAT64,
        Optimum_Stock_level_Qty FLOAT64,
        Stock_Change_Signal STRING,
    
    ) 
    
    Always convert Location ID, Part ID, Part Number, Part Age, PrimaryVolumeType, Model Name, Stock Level Qty.
     
    To show the quantity level of Forecast, always return with and return with Part ID, Optimum Stock Level Qty and the query directly.
    
    Always give a simple alias name to a column if any operation has been performed on that column.
    
    Always consider inventory as inventory_on_hand.
    
    Consider Inventory_on_hand as Inventory on hand. 
    
    Consider Qty as Quantity. 

    Consider Date as Month/Day/Year format or MM/DD/YYYY. 

    Consider period as Forecast Priod ID.
       
    Consider all the business months and don't do any partition.
    
    Remember that before you answer a question, you must check to see if it compiles with your mission above.

    Question : What are the part numbers for which forecast quantity smaller than optimum in period 8/1/2024.
    
    Answer : SELECT Part_Number, Forecast_Quantity, Forecast_Period_ID 
                FROM `ds.inventory`  
                WHERE (Forecast_Period_ID = '2024-08-01') AND (Forecast_Quantity < Optimum_Stock_level_Qty) 

    Question : What are the part numbers for which forecast quantity greater than optimum in period 8/1/2024.
    
    Answer : Answer : SELECT Part_Number, Forecast_Quantity, Forecast_Period_ID 
                FROM `ds.inventory`  
                WHERE (Forecast_Period_ID = '2024-08-01') AND (Forecast_Quantity > Optimum_Stock_level_Qty) 

    Lets think step by step and return only the sql query.
          
    ''', "us-central1")

    #dict_sql = str(data.to_dict())

    #print(data)
    
    return data


# In[87]:


question="What are the part numbers for which forecast quantity smaller than optimum in period 8/1/2024? "
question=question.lower()
if "period" in question:
    question=question+" and show the Forecast Period ID"
if "optimum" in question:
    question=question+" and show the Optimum_Stock_level_Qty"
if "inventory" in question:
    question=question+" and show the Inventory_on_hand"
if "Part Number" in question:
    question=question+" and show the Part ID"
if "how many" in question:
    question=question+" and show count"
print(question)
result = main_fun(question)
print(result)


# In[88]:


def run_sql_query(sql):
    # Create a BigQuery client

    client = bigquery.Client()

        # Get the list of tables in the dataset

    tables = client.list_tables('ds')

    results = client.query(sql).to_dataframe()

    return results


# In[89]:


result1=run_sql_query(result)


# In[90]:


print(result1)


# In[91]:


parameters = {

    "temperature": 0,

    "max_output_tokens": 1024,


}


# In[92]:


def img_to_html1(img_path, ht = 500, wd = 1000):
    img_html = "<img src='data:image/png;base64,{}' height='{}' width='{}'>".format(
        img_to_bytes(img_path),ht,wd
    )
    return img_html


# In[93]:


@retry(tries=2)
def table_to_text(question,ans):
    model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
    instruction = """ Given a table and a question. 
    
    convert the ans table to a human readable text sentence according to the question.
    
    """
    result=model.predict(f'''{instruction},
                    question:{question},
                     ans:{ans} 
                     ''',**parameters)
    data=result.text
    return data


# In[94]:


#sentence1=table_to_text(question,result1)
#print(sentence1)


# In[95]:


#result1.to_csv('data.csv')
#result1.to_csv('ans.csv')


# In[96]:


@retry(tries=2)
def plot_code(question,ans):
    ans.to_csv('data.csv')
    ans.to_csv('ans.csv')
    ans.to_csv('table.csv')
    model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
    instruction = """ Given a table load it into a python dataframe named 'df'.
                      Generate line graph for integer or float values and bar graph for string values.
                      Generate a python code using plotly to plot the df appropriately in a graph and give appropriate title and resize the graph according to the values.
                      y axis or x axis can have multiple values. All columns should present in the graph.
    """
    result=model.predict(f'''{instruction},
                    ques:{question},
                     ans:{ans} ''',**parameters)
    return result


# In[ ]:





# In[97]:


@st.cache_data
def graph_plot(graph,question,results):
    if graph:
        if len(results)==1 or len(results.columns)==1:
            st.write('Graph not available')
        else:
            try:
                if 'Forecast_Period_ID' in results.columns:
                    results=results.sort_values('Forecast_Period_ID')
                plot_py=str(plot_code(question,results))
                plot_py=plot_py.replace('```','').replace('python','').replace('fig.show()','st.plotly_chart(fig)') 
                results.to_csv('data.csv')
                results.to_csv('ans.csv')
                results.to_csv('table.csv')
                exec(plot_py)
            except:
                st.write('Graph not available')


# In[ ]:


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


# In[ ]:


def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' height='35' width='45'>".format(
        img_to_bytes(img_path)
    )
    return img_html


# In[ ]:


def main():
    st.set_page_config(page_title = "Toyota Data Core",layout="wide")
    padding_top = 0
    st.markdown(
        f"""
    <style>
        .appview-container .main .block-container {{
            padding-top: 0;
            margin: 0;
            height: 98%;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.text("")

    st.markdown(
        """
    <style>
        .background {{
            background-color: rgb(241, 237, 238);
            padding: 10px;
            margin-top: 1%;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
        }}

        .title_heading {{
            color: #000000;
            font-size: 22px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
        }}

        .title {{
            margin-top: 20px;
            display: flex;
        }}

        .button-inline {{
            color: green;
            background-color: rgb(241, 237, 238);
            padding: 10px 20px;
            font-size: 11px;
            font-weight: bold;
            border: 1px solid white;
            margin-left: auto;
            margin-right: 10px;
            height: 20px;
            margin-top: 1px;
            line-height: 0.3;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }}

        .vertical-bar {{
            display: inline-block;
            height: 1em;
            vertical-align: middle;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        .background_black {{
            background-color: #000000;
            padding-top: 0px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            margin-top: 2%;
            margin-bottom: -3%;
            position: relative;
        }}

        .paragraph_heading {{
            color: rgb(134, 188, 37);
            font-size: 18px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
        }}

        .paragraph_body {{
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
        }}

        .paragraph {{
            margin-left: 20px;
            margin-top: 10px;
        }}

        .image {{
            position: absolute;
            top: 8;
            right: 0;
            margin-left: 10px;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        div.css-1vbkxwb.eqr7zpz4 {{
            color: green;
            margin-top: 10%;
            text-align: center;
        }}

        .css-1vbkxwb.eqr7zpz4 p {{
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: bold;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        button.css-1n543e5.e1ewe7hr5 {{
            padding: 2px 2px 2px 2px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            height: 60%;
            width: 6%;
            text-align: center;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        div.css-12ttj6m.en8akda1 {{
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        input.st-be.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-c7.st-c8.st-b8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-cf.st-ae.st-af.st-ag.st-ch.st-ai.st-aj.st-by.st-ci.st-cj.st-ck {{
            background-color: rgb(241, 237, 238);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        input.st-bd.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-c7.st-b8.st-c8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-cf.st-ae.st-af.st-ag.st-cg.st-ai.st-aj.st-bx.st-ch.st-ci.st-cj {{
            background-color: rgb(241, 237, 238);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        textarea.st-bd.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-c7.st-b8.st-c8.st-c9.st-ca.st-cb.st-cp.st-cq.st-cr.st-cs.st-ae.st-af.st-ag.st-cg.st-ai.st-aj.st-bx.st-ch.st-ci.st-cj.st-ct.st-cu.st-cv {{
            background-color: rgb(241, 237, 238);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


    # Define a CSS style for the buttons
    button_style = """
        <style>
            .equal-width-button button {
                width: 200px;
                box-sizing: border-box;
            }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


    st.markdown("""
        <style>
        
            .background {
            background-color: rgb(241, 237, 238);
            padding: 10px;
            margin-top: -75px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            }
            
            .title_heading {
            color: #000000;
            font-size: 22px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
            }
            .title {
            margin-top: 20px;
            display: flex;
            }
            .button-inline {
            color: green;
            background-color: rgb(241, 237, 238);
            padding: 10px 20px;
            font-size: 11px;
            font-weight: bold;
            border: 1px solid white;
            margin-left: auto;
            margin-right: 10px;
            height: 20px;
            margin-top: 1px;
            line-height: 0.3;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
            }

            .vertical-bar {
            display: inline-block;
            height: 1em;
            vertical-align: middle;
        }
            
            """
            
        f"""</style>
        <div class="background">
            <p class="title">
            {img_to_html('toyota.jpg')}
            <span class ="title_heading"> | Generative AI</span>
            <button class="button-inline" type="button">Logout</button>
        </p>
        </div>
            """,

                unsafe_allow_html=True,

                )
    ##Create a text container with a black background
    st.markdown("""
        <style>
        
            .background_black {
            background-color: #000000;
            padding-top: 0px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            margin-top: 2%;
            margin-bottom: -3%;
            position: relative;
            }
            
            .paragraph_heading {
            color: rgb(134, 188, 37);
            font-size: 18px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
            }
            
            .paragraph_body {
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
            }
            .paragraph {
            margin-left: 20px;
            margin-top: 10px;
            }
            .image{
            position: absolute;
            top: 8;
            right: 0;
            margin-left: 10px;
            }

            
        </style>
        <div class="background_black">
        <p class="paragraph">
            <span class ="paragraph_heading">Tabular Question Answering</span><br>
            <span class ="paragraph_body">A generative AI powered tool which can efficiently answer questions from tables.</span>
            
        </p>
        </div>
            """,

                unsafe_allow_html=True,

                )
    st.markdown("---")

    if 'key' not in st.session_state:
        st.session_state.key = False

    if 'clear_cache' not in st.session_state:
        st.session_state.clear_cache = False
    c1,c2 = st.columns([6,1])
    with c1:
        question = st.text_input("Ask your question related to the Forecast and Inventory")
        question=question.lower()
    with c2:   
        st.markdown("#")
        generate_response = st.button("Submit")
    graph=st.checkbox('Graph')
    if question and generate_response:
        st.session_state.clear_cache = True

    if (generate_response or st.session_state.key):  
        #if st.session_state.clear_cache:
        #   main_fun.clear()
        #    run_sql_query.clear()
        try:
            response = main_fun(ques=question)
            if "period" in response:
                response=response.replace("period", "Forecast_Period_ID")
            if "finance_hold_indicator" in response:
                response=response.replace("optimum", "Optimum_Stock_level_Qty")
            if "damage_hold_indicator" in response:
                response=response.replace("inventory", "Inventory_on_hand")
            
            def split_sentence_from_word(sentence, target_word):
                # Find the index of the target word in the sentence
                target_index = sentence.find(target_word)

                # If the target word is not found, return None
                if target_index == -1:
                    return None

                # Split the sentence into two parts: before and after the target word
                before_word = sentence[:target_index].strip()
                after_word = sentence[target_index + len(target_word):].strip()

                return before_word, after_word
            sentence = response
            target_word = "FROM"
            words = split_sentence_from_word(sentence, target_word)
            if words:
                before_word, after_word = words
            else:
                st.write("Target word not found in the sentence.")

            def remove_lines_before_group_by(input_string):
                lines = input_string.split('\n')
                group_by_found = False
                result_lines = []

                for line in lines:
                    if re.match(r'^\s*GROUP\s+BY', line, re.IGNORECASE):
                        group_by_found = True
                        result_lines.append(line)
                    elif group_by_found:
                        result_lines.append(line)

                result_string = '\n'.join(result_lines)
                return result_string

            # Example input string
            input_string = after_word

            output_string = remove_lines_before_group_by(input_string)
            #if ('region' in question) and ('region' not in before_word) and ('region' in output_string):
            #    before_word=before_word.replace('SELECT', 'SELECT region,')
            #if ('dealer' in question) and ('dealerName' not in before_word) and ('dealerName' in output_string):
            #    before_word=before_word.replace('SELECT', 'SELECT dealerName,')
            
            before_word=before_word+" FROM "
            final_query=before_word + after_word
            results=run_sql_query(final_query)
            st.markdown("---")
            if len(results)!=1 or len(results.columns)!=1:
                st.write(results)
            col=results.columns
            if (results.shape[0]==1 and len(col)==2) or len(col)==1:
                result2=table_to_text(question,results) 
                st.write(result2)
            st.session_state.key = True
            if st.button(':bulb:'):
                st.write(final_query)

            graph_plot(graph,question,results)
        except Exception as e:
                    st.info(f'Could not generate an answer because of the error : {e}')
    
    st.session_state.clear_cache = False

if __name__ == "__main__":
    main()  


# In[ ]:





# In[ ]:





# In[ ]:




