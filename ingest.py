import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load CSV
df = pd.read_csv("data/online_courses.csv")

# Fill missing values
df = df.fillna("")

# Convert ALL columns to string (important fix)
df = df.astype(str)

# Combine important fields
text_data = (
    "Course Title: " + df["Title"] + ". " +
    "Category: " + df["Category"] + ". " +
    "Sub Category: " + df["Sub-Category"] + ". " +
    "Level: " + df["Level"] + ". " +
    "Duration: " + df["Duration"] + ". " +
    "Instructor: " + df["Instructors"] + ". " +
    "Skills: " + df["Skills"] + ". " +
    "Description: " + df["Short Intro"] + ". " +
    "What you learn: " + df["What you learn"] + ". " +
    "Price: " + df["Price"] + "."
)

docs = text_data.tolist()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

chunks = splitter.create_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print(f"Indexed {len(chunks)} chunks successfully!")