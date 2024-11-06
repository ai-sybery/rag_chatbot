import streamlit as st
from models import RAGBot
from utils import DocumentProcessor
import os
import tempfile

def initialize_bot():
    """Инициализация компонентов в session_state"""
    if 'bot' not in st.session_state:
        try:
            st.session_state.bot = RAGBot()
        except Exception as e:
            st.error(f"Ошибка инициализации бота: {str(e)}")
            return False
            
    if 'processor' not in st.session_state:
        try:
            st.session_state.processor = DocumentProcessor()
        except Exception as e:
            st.error(f"Ошибка инициализации процессора документов: {str(e)}")
            return False
            
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    return True

def handle_file_upload(uploaded_file):
    """Обработка загруженного файла"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        docs = st.session_state.processor.load_document(tmp_path)
        st.session_state.bot.vector_store = st.session_state.processor.create_vector_store(docs)
        
        os.unlink(tmp_path)  # Удаляем временный файл
        return True
        
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
        return False

def main():
    st.title("RAG Chat Bot")
    
    if not initialize_bot():
        st.error("Ошибка инициализации приложения")
        return

    # Сайдбар для загрузки документов
    with st.sidebar:
        st.header("Загрузка документов")
        uploaded_file = st.file_uploader("Выберите документ", type=['pdf', 'txt'])
        
        if uploaded_file:
            with st.spinner("Обработка документа..."):
                if handle_file_upload(uploaded_file):
                    st.success("Документ успешно загружен!")

    # Отображение истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Обработка нового сообщения
    if prompt := st.chat_input("Задайте вопрос"):
        # Добавляем вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Получаем и отображаем ответ
        with st.chat_message("assistant"):
            try:
                with st.spinner("Генерация ответа..."):
                    response = st.session_state.bot.get_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Ошибка при получении ответа: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()