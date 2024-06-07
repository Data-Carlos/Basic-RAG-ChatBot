# Tutorial: Building a Knowledge-Enhanced Chatbot with LangChain and Pinecone

## Introduction

Welcome to this comprehensive tutorial on creating a knowledge-enhanced chatbot using LangChain and Pinecone. In this tutorial, we will guide you through the process of building a chatbot capable of utilizing a knowledge base to enhance its responses. The integration of LangChain and Pinecone allows us to efficiently manage and query a vast amount of information, enabling our chatbot to provide more accurate and contextually relevant answers.

## Objective

By the end of this tutorial, you will be able to:
1. Set up and initialize a Pinecone vector index.
2. Embed text data using OpenAI's embedding models.
3. Populate the vector index with embedded data.
4. Perform similarity searches on the vector index.
5. Integrate the knowledge base with a chatbot to augment its responses.

## Audience

This tutorial is intended for intermediate-level developers who have a basic understanding of Python programming, machine learning concepts, and familiarity with APIs. Some experience with chatbot development and working with APIs will be beneficial.

## Resources

Before you start, ensure you have the following:
- Python 3.7 or higher installed on your system.
- An OpenAI API key.
- A Pinecone API key.
- The dataset of Arvix papers (`chunks_articles.csv`).

Install the required libraries using pip:
```sh
pip install -qU langchain==0.0.354 openai==1.6.1 datasets==2.10.1 pinecone-client==3.1.0 tiktoken==0.5.2
```

## Structure

1. **Introduction and Setup**
2. **Loading and Preparing the Dataset**
3. **Building the Knowledge Base**
4. **Creating Embeddings and Populating the Index**
5. **Retrieval Augmented Generation (RAG)**
6. **Integrating with the Chatbot**
7. **Conclusion**

## 1. Introduction and Setup

To begin, we'll create a simple chatbot without any retrieval augmentation by initializing a `ChatOpenAI` object. Ensure you have your OpenAI API key ready.

## 2. Loading and Preparing the Dataset

Load the dataset of Arvix papers from a local CSV file (see file above) using pandas:

```python
import pandas as pd

# Load the dataset
dataset = pd.read_csv("chunks_articles.csv")

# Display the first few rows of the dataset
dataset.head()
```

## 3. Building the Knowledge Base

### Initializing Pinecone

Set up your Pinecone API key and initialize the Pinecone client:

```python
from pinecone import Pinecone
import os

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or "your_pinecone_api_key"

# configure client
pc = Pinecone(api_key=api_key)
```

### Setting Up the Index Specification

Configure the cloud provider and region for your index:

```python
from pinecone import ServerlessSpec

spec = ServerlessSpec(cloud="aws", region="us-east-1")
```

### Initializing the Index

Create and initialize the index if it doesn't already exist:

```python
import time

index_name = 'rag'
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# check if index already exists
if index_name not in existing_indexes:
    # create index
    pc.create_index(index_name, dimension=1536, metric='dotproduct', spec=spec)
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()
```

## 4. Creating Embeddings and Populating the Index

### Instantiating the Embeddings Model

Set up OpenAI's embedding model via LangChain:

```python
from langchain.embeddings import OpenAIEmbeddings
import os

# Ensure you have your OpenAI API key set in your environment variables
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Instantiate the embeddings model
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
```

### Embedding Text Data

Generate embeddings for sample text data:

```python
texts = ['this is the first chunk of text', 'then another second chunk of text is here']

res = embed_model.embed_documents(texts)
len(res), len(res[0])
```

### Embedding and Indexing the Dataset

Embed and insert data into Pinecone in batches:

```python
from tqdm.auto import tqdm
import pandas as pd

data = dataset  # this makes it easier to iterate over the dataset
batch_size = 30

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i + batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['id']}-{x['chunk_id']}" for _, x in batch.iterrows()]
    texts = [str(x['chunk']) for _, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    metadata = [{'text': str(x['chunk']), 'source': str(x['authors']), 'title': str(x['title'])} for _, x in batch.iterrows()]
    index.upsert(vectors=list(zip(ids, embeds, metadata)))
```

## 5. Retrieval Augmented Generation (RAG)

### Initializing the Vector Store

Set up LangChain's vectorstore with our Pinecone index:

```python
from langchain.vectorstores import Pinecone

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(index, embed_model.embed_query, text_field)
```

### Querying the Index

Perform a similarity search to retrieve relevant information:

```python
query = "What is LLAMA3?"
vectorstore.similarity_search(query, k=3)
```

### Augmenting the Prompt

Create a function to augment the chatbot's prompt with retrieved information:

```python
def augment_prompt(query: str):
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

# Example usage
print(augment_prompt(query))
```

## 6. Integrating with the Chatbot

Connect the augmented prompt to the chatbot:

```python
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are an expert in the field of AI."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand Recursice Neural Networks.")
]
chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')

# create a new user prompt
prompt = HumanMessage(content=augment_prompt(query))
messages.append(prompt)

res = chat(messages)
print(res.content)

# Asking the same question with no RAG
prompt = HumanMessage(content="What is LLAMA3?")
res = chat(messages + [prompt])
print(res.content)
```

## Conclusion

By following this tutorial, you've learned how to build a knowledge-enhanced chatbot that leverages a robust knowledge base using LangChain and Pinecone. This setup allows the chatbot to provide more accurate and contextually relevant responses by retrieving and integrating information from the knowledge base.

Experiment with different datasets, queries, and embeddings to further enhance your chatbot's capabilities. Happy coding!
