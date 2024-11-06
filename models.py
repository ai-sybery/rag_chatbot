from langchain.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
import google.generativeai as genai
from config import get_gemini_model, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class RAGBot:
    def __init__(self):
        self.model = get_gemini_model()
        try:
            self.graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
        except Exception as e:
            print(f"Ошибка подключения к Neo4j: {str(e)}")
            self.graph = None
            
        self.vector_store = None
        self.chat_session = None

    def init_chat(self):
        try:
            self.chat_session = self.model.start_chat(history=[])
        except Exception as e:
            raise Exception(f"Ошибка инициализации чата: {str(e)}")
        
    def get_response(self, query):
        if not self.chat_session:
            self.init_chat()
            
        try:
            # Получаем контекст из графа и векторного хранилища
            graph_context = self.get_graph_context(query) if self.graph else []
            vector_context = self.get_vector_context(query)
            
            # Формируем промпт с контекстом
            context = []
            if graph_context:
                context.append(f"Graph information: {graph_context}")
            if vector_context:
                context.append(f"Vector search results: {vector_context}")
            
            prompt = f"""Context from knowledge base:
            {' '.join(context)}
            
            Question: {query}"""
            
            response = self.chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            return f"Ошибка при получении ответа: {str(e)}"

    def get_graph_context(self, query):
        if not self.graph:
            return []
        try:
            cypher_query = f"""
            MATCH (n)
            WHERE n.content CONTAINS '{query}'
            RETURN n.content LIMIT 3
            """
            result = self.graph.query(cypher_query)
            return result
        except Exception as e:
            print(f"Ошибка при поиске в графе: {str(e)}")
            return []

    def get_vector_context(self, query):
        if not self.vector_store:
            return []
        try:
            results = self.vector_store.similarity_search(query, k=3)
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Ошибка при векторном поиске: {str(e)}")
            return []