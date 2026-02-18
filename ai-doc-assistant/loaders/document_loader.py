from langchain_community.document_loaders import PyPDFLoader
def load_pdf():
    loader = PyPDFLoader(file_path)
    return loader.load()

