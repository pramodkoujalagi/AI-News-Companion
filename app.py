import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import initialize_agent, Tool
from langchain.prompts import MessagesPlaceholder
from langchain.chains.summarize import load_summarize_chain
import requests
import json
import streamlit as st
from langchain.schema import SystemMessage

load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': SERPAPI_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url,
                                headers=headers,
                                data=payload)
    response_data = response.json()
    
    print("search results:", response_data)
    return response_data


#2 llm to choose the best articles, and return urls
def find_best_article_urls(response_data, query):
    #turn json into string
    response_str = json.dumps(response_data)
    
    #creating llm to choose best articles
    template = """
    You are a world class jounalist & researcher and news searcher, you are extremely good at finding the most relevant articles to certain topic;
    {response_str}
    Above is the list of search results for the query {query}.
    Please choose the best 3 articles from the list, return ONLY an array of the urls, do not include anything else; return ONLY an array of the urls, do not include anything else other than urls;
    """
    
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"], template=template)
    article_picker_chain = LLMChain(
        llm=OpenAI(), prompt=prompt_template, verbose=True)
    
    urls = article_picker_chain.predict(response_str=response_str, query=query)
    
    #Convert string to list
    url_list = json.loads(urls)
    print(url_list)
    
    return url_list

#3 get content for each article from urls and make summaries

def get_content_from_urls(urls):
    #using unstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    return data

def summarise(data, query):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len)
    text = text_splitter.split_documents(data)
    
    template = """
    You are a world class journalist, you are extremely good at summarising and you will try to summarise the text above in order to create a better summary about {query}
    Please follow all of the following rules:
    1/ Make sure the answer is direct and concise, not more than 100 words
    2/ Make sure the content is not too long, informative with good data
    3/ The content should address the {query} topic very well
    4/ The content should be written in a way that is easy to read and understand
    
    SUMMARY:
    """
    
    prompt_template = PromptTemplate(
        input_variables=["text", "query"], template=template)
    summariser_chain = LLMChain(
        llm=OpenAI(), prompt=prompt_template, verbose=True)
    summaries = []
    
    for chunk in text:
        summary = summariser_chain.predict(text=chunk, query=query, articles=query)
        summaries.append(summary)
        
    print(summaries)
    return summaries

def generate_news(summaries, query):
    summaries_str = str(summaries)
    
    template = """
    {summaries_str}
    You are a world class jounalist and powerful socially, text above is some context about{query}
    Please follow all of the following rules:
    1/ The thread needs to be direct and concise, not more than 100 words
    2/ The thread needs to be not too long, informative with good data
    3/ The thread needs to address the {query} topic very well
    5/ The thread needs to be written in a way that is easy to read and understand
    
    News:
    """
    
    prompt_template = PromptTemplate(
        input_variables=["summaries_str", "query"], template=template)
    news_chain = LLMChain(
        llm=OpenAI(), prompt=prompt_template, verbose=True)
    
    news = news_chain.predict(summaries_str=summaries_str, query=query)
    
    return news


def main():
    st.set_page_config(page_title="Epic Newzz!", page_icon="🌏", layout="wide")
    
    st.header("Hi👋 I am Newzy from Epic Newzz.. Ask me anything!😁")
    query = st.text_input("News Topic")
    
    if query:
        print(query)
        st.write("Gathering news for:", query)
        
        search_results = search(query)
        urls = find_best_article_urls(search_results, query)
        data = get_content_from_urls(urls)
        summaries = summarise(data, query)
        news = generate_news(summaries, query)
        
        with st.expander("search results"):
            st.info(search_results)
        with st.expander("links to relevant articles"):
            st.info(urls)
        with st.expander("news"):
            st.info(news)


if __name__ == '__main__':
    main()