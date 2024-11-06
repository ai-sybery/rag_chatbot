import os
from typing import List
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            raise Exception(f"Ошибка инициализации embeddings: {str(e)}")

    def load_document(self, file_path: str) -> List:
        """Загрузка документа в зависимости от типа файла"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
            
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                raise ValueError("Неподдерживаемый тип файла")
            
            documents = loader.load()
            if not documents:
                raise ValueError("Документ пуст или не может быть прочитан")
                
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise Exception(f"Ошибка при загрузке документа: {str(e)}")

    def create_vector_store(self, documents: List, store_name: str = "vector_store"):
        """Создание векторного хранилища"""
        if not documents:
            raise ValueError("Документы для индексации не предоставлены")
            
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            # Создаем директорию если её нет
            os.makedirs(store_name, exist_ok=True)
            vector_store.save_local(store_name)
            return vector_store
        except Exception as e:
            raise Exception(f"Ошибка создания векторного хранилища: {str(e)}")

    def load_vector_store(self, store_name: str = "vector_store"):
        """Загрузка существующего векторного хранилища"""
        try:
            if os.path.exists(store_name):
                return FAISS.load_local(store_name, self.embeddings)
            return None
        except Exception as e:
            raise Exception(f"Ошибка загрузки векторного хранилища: {str(e)}")