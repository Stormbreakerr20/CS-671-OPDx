from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import Dict,Any

from langchain_community.chat_models import ChatOllama
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

import os
import uuid
import subprocess
import torch
import base64
import time
import glob

#!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen("ollama serve", shell=True)

#!sleep 10
subprocess.Popen("cd ./../../Model && ollama create opdx -f ./Modelfile", shell=True)
time.sleep(10)
subprocess.Popen("ollama pull llava:7b-v1.5-q4_0", shell=True)
time.sleep(10)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000"
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=HuggingFaceEmbeddings(),
)

store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

llm = Ollama(model="opdx", stop = ["###", "{", "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."], system = "If I tell you that I am not well or have any medical problem, analyse my conditions carefully, and remember any information about my symptoms or medical history. Keep asking follow-up questions about the symptoms and any other information that may help you to narrow down at a differential diagnosis. Do not ask more than 2 questions at a time. If and only if you are confident about it, provide me with a list of possible diagnoses, three or four at maximum, ranked by likelihood, and a brief explanation of your reasoning. Keep asking questions otherwise, not more than 2 at a time.",num_gpu=1)
llava = Ollama(model="llava:7b-v1.5-q4_0")
print(llava("Hi"))

def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/upload_img")
async def up_img(inp)-> Dict[str,str]:
    try:
        cleaned_img_summary = []
        filename = inp
        
    # Iterate over matched file paths
        for img in [filename]:
            RES = llava(prompt="Provide a concise, factual summary of the image, capturing all the key visual elements and details you observe. Avoid speculative or imaginative descriptions not directly supported by the contents of the image. Focus on objectively describing what is present in the image without introducing any external information or ideas. Your summary should be grounded solely in the visual information provided." ,images=[str(image_to_base64(f"./{img}"))])
            cleaned_img_summary.append(RES)
        # print(cleaned_img_summary)
        
        img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(cleaned_img_summary)
        ]

        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(
            list(zip(img_ids, cleaned_img_summary))
        )
        
        return {"message":"200"}
    
    except Exception as e: 
        print(e)
        return {"message":"404"}

@app.post("/generate")
async def generate_output(inp: str) -> Dict[str, str]:
    s = "You Are my personal doctor. You have to Remember my symptoms antecendants past history and try to ask me follow up questions whenever I tell you that I am not well"

# Prompt template
    template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. {{ if .}}### Instruction: If I tell you that I am not well or have any medical problem, analyse my conditions carefully, and remember any information about my symptoms or medical history. Keep asking follow-up questions about the symptoms and any other information that may help you to narrow down at a differential diagnosis. Do not ask more than 2 questions at a time. If and only if you are confident about it, provide me with a list of possible diagnoses, three or four at maximum, ranked by likelihood, and a brief explanation of your reasoning. Keep asking questions otherwise, not more than 2 at a time. You also have the following context, which can include text and tables to answer my QUESTION {{ .System }}{{ end }} {{ if .Prompt }}{context}
                    Question: {question}{{ .Prompt }}{{ end }} ### Response:"""
    prompt = ChatPromptTemplate.from_template(template)

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    txt = chain.invoke(inp)

    return {"message":txt}

@app.post("/upload")
async def post_pdf() -> Dict[str,str]:
    try:
        print("here1")
        filename = "uploaded_file.pdf"
        path = "./"

        # Extract images, tables, and chunk text
        raw_pdf_elements = partition_pdf(
            filename = path + filename,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=path,
        )
        category_counts = {}
        
        print("Safe1")

        for element in raw_pdf_elements:
            category = str(type(element))
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

        # Unique_categories will have unique elements
        # TableChunk if Table > max chars set above
        unique_categories = set(category_counts.keys())
        category_counts
        class Element(BaseModel):
            type: str
            text: Any

        # Categorize by type
        categorized_elements = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                categorized_elements.append(Element(type="table", text=str(element)))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                categorized_elements.append(Element(type="text", text=str(element)))

        print("safe2")
        # Tables
        table_elements = [e for e in categorized_elements if e.type == "table"]
        # print(len(table_elements))

        # Text
        text_elements = [e for e in categorized_elements if e.type == "text"]
        # print(len(text_elements))

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text. \
        Give a concise summary of the table or text. Table or text chunk: {element} """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

        texts = [i.text for i in text_elements if i.text != ""]
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        print(text_summaries)
        # Apply to tables
        tables = [i.text for i in table_elements]
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

        # Add texts
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

        # Add tables
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=s, metadata={id_key: table_ids[i]})
            for i, s in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

        subprocess.Popen("ollama serve", shell=True)
        #!sleep 10

        IMG_DIR = './'

        # Use glob to match file paths
        image_files = glob.glob(f"{IMG_DIR}*.jpg")
        # print(image_files)
        cleaned_img_summary = []

        # Iterate over matched file paths
        for img in image_files:
            # Perform your operation here
            RES = llava(prompt="Provide a concise, factual summary of the image, capturing all the key visual elements and details you observe. Avoid speculative or imaginative descriptions not directly supported by the contents of the image. Focus on objectively describing what is present in the image without introducing any external information or ideas. Your summary should be grounded solely in the visual information provided." ,images=[str(image_to_base64(img))])
            cleaned_img_summary.append(RES)
            
        img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(cleaned_img_summary)
        ]
        print(cleaned_img_summary)
        
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(
            list(zip(img_ids, cleaned_img_summary))
        )
        return {"message":"200"}
    except Exception as e:
        print(e)
        return {"message":"404"}