import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# Problemli chains modulunun əvəzinə müasir LCEL importları istifadə edirik:
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sən görüşləri analiz edən köməkçi AI-san. Aşağıdakı kontekstdən istifadə edərək istifadəçinin sualına cavab ver. Əgər cavab kontekstdə yoxdursa, uydurma və 'Bu barədə məlumat yoxdur' de.\n\nKontekst:\n{context}"),
            ("human", "{input}")
        ])
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Tapılan mətnləri birləşdirmək üçün kiçik funksiya
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Heç bir chains importu olmadan müasir Zəncir (LCEL) qururuq
        self.qa_chain = (
            {"context": self.retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def add_transcript(self, text: str):
        """Transkripti hissələrə bölüb ChromaDB-yə əlavə edir."""
        if not text.strip():
            return
        
        docs = [Document(page_content=text)]
        splits = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)

    def ask(self, question: str) -> str:
        """Görüşün mətni üzrə sual verir."""
        # Köhnə versiyadan fərqli olaraq, birbaşa string cavab qayıdır
        return self.qa_chain.invoke(question)

# Modulu test etmək üçün
if __name__ == "__main__":
    rag = RAGEngine()
    
    sample_meeting = """
    Bugünkü görüşdə qərara gəldik ki, serverləri AWS-dən qaldıraq. 
    Büdcə olaraq aylıq 500 dollar ayrıldı. 
    Marketinq komandası isə yeni kampaniyaya gələn ayın 15-də başlayacaq və 
    sosial mediada video reklamlar üzərində fokuslanacaqlar.
    """
    
    print("Mətn vektor bazasına (ChromaDB) əlavə edilir...")
    rag.add_transcript(sample_meeting)
    print("Mətn əlavə olundu!\n")
    
    question = "Marketinq komandası kampaniyaya nə vaxt başlayır və büdcə nə qədərdir?"
    print(f"Sual: {question}")
    
    try:
        answer = rag.ask(question)
        print(f"Cavab: {answer}")
    except Exception as e:
        print("Xəta baş verdi:", e)