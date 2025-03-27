from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings


import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

print("# Load and index knowledge base files")
knowledge_base_files = ["./products.txt", "./policies.txt"]


def load_data():
    documents = []
    for file in knowledge_base_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip().split("\n\n")  # Split sections
            for section in text:
                documents.append(section)
    return documents

knowledge_data = load_data()

print("#Load Embeddings")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("# feed data chunks into Chroma Vector DB")
persist_directory = "./chroma_db"

vectordb = Chroma.from_texts(
    texts=knowledge_data, embedding=embeddings, persist_directory=persist_directory
)

print("#LLM for QnA bot")
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

LLM_Model = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(LLM_Model)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_Model, torch_dtype=torch.float32)

print("# Create a huggingface pipeline with the llm")

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer,
    min_new_tokens=5,    # Ensures a minimum response length  
    max_length=200,  # Ensures longer responses  
    temperature=0.4,     # Reduces randomness for definitive answers  
    top_p=0.5,           # Balances diversity without too much randomness  
    top_k=40,            # Limits token choices for controlled output  
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

print("# Create a prompt for the input question, this prompt will be fed to LLM along with context")

from langchain.prompts import PromptTemplate                #prompt
from langchain.memory import ConversationBufferMemory       #memory
from langchain.chains import RetrievalQA                    #chain

prompt_template = """
You are an intelligent assistant that answers user questions accurately based on the given data.  
Respond concisely but completely, without adding extra information.  
If the question is not covered in the provided data, respond with "I'm sorry, but I don't have that information."

Examples:
Q: What is the battery life of the MacBook Air M2?  
A: The MacBook Air M2 has a battery life of up to 18 hours.

Q: How much does express shipping cost?  
A: Express shipping costs $14.99 and takes 2-3 business days.

Q: When can I get my refund?  
A: Refunds are processed within 7 business days after the return is received.

Now answer this question based on the given context:  

Context: {context}
question: {question}
answer:"""
prompt = PromptTemplate(template=prompt_template)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,  # Return messages for better context
)


print("#Finally initializing the Retrieval QnA chain")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Use "stuff" for simplicity
    retriever=vectordb.as_retriever(search_type="similarity",  search_kwargs={"k": 3}),
    verbose=True,       # Print debugging info
    memory=memory,
    chain_type_kwargs={
        "prompt": prompt
    }
)

print("#Run the Chatbot")
while True:
    # Get user input
    print("Please type your question or type exit/quit to quit")
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break

    # Generate a response
    try:
        # Format the input as a dictionary with the key "query"
        #input_dict = {"query": question}
        response = qa({"query":question})
        print(f"Bot: {response['result']}+\n")  # Extract the answer from the response
    except Exception as e:
        print(f"Error: {e}")
