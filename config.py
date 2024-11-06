import os
import google.generativeai as genai
from dotenv import load_dotenv

# Загружаем переменные из .env если есть
load_dotenv()

# Пробуем получить из разных источников
def get_env_var(var_name: str) -> str:
    # Сначала из .env или переменных окружения
    value = os.getenv(var_name)
    if value:
        return value
    # Затем из Streamlit secrets если доступно
    try:
        import streamlit as st
        return st.secrets[var_name]
    except:
        return ""

# Конфигурация Gemini
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Получаем переменные
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
NEO4J_URI = get_env_var("NEO4J_URI")
NEO4J_USER = get_env_var("NEO4J_USER")
NEO4J_PASSWORD = get_env_var("NEO4J_PASSWORD")

# Инициализация Gemini
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_model():
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        generation_config=GENERATION_CONFIG,
        system_instruction="ассистент"
    )