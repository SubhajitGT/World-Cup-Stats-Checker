from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from typing import List, Tuple, Type
import streamlit as st

# load the document and split it into chunks
def qnA():
    st.title("Cricket-Football World Cup Stats Chat App")
    path1 = "cricket.txt"
    path2 = "football.txt"

    options = ["CRICKET", "FOOTBALL"]

    # Create a select box
    selected_option = st.selectbox("Choose an option:", options)

    # Display the selected option
    st.write(f"You selected: {selected_option}")

    if selected_option == 'CRICKET':
        loader1 = UnstructuredFileLoader(
            path1,
            strategy="fast",
            mode="paged"
        )
    elif selected_option == 'FOOTBALL':
        loader1 = UnstructuredFileLoader(
            path2,
            strategy="fast",
            mode="paged"
        )
    documents = loader1.load()

# split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

# create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    docs = filter_complex_metadata(docs)

# load it into Chroma
    db = Chroma.from_documents(docs, embedding_function)

# query it
    query = st.text_input("Question: ")
    #index = VectorstoreIndexCreator().from_loaders([loader1])
    #print(index.query(query))
    docs = db.similarity_search(query)

# print results
    st.write(docs[0].page_content)

def filter_complex_metadata(documents,
    *,
    allowed_types: Tuple[Type, ...] = (str, bool, int, float)):
    """Filter out metadata types that are not supported for a vector store."""
    updated_documents = []
    for document in documents:
        filtered_metadata = {}
        for key, value in document.metadata.items():
            if not isinstance(value, allowed_types):
                continue
            filtered_metadata[key] = value

        document.metadata = filtered_metadata
        updated_documents.append(document)

    return updated_documents

if __name__ == "__main__":
    qnA()