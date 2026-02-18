from langchain_community.document_loaders import PyPDFLoader
def load_pdf():
    loader = PyPDFLoader(r"C:\Users\Yoni\OneDrive\Desktop\Langchain  AI Chatbot\Database Recovery Techniques.pdf")
    return loader.load()

