import os
import google.generativeai as genai
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Проверяем наличие необходимых переменных окружения
required_env_vars = ["GEMINI_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Отсутствуют необходимые переменные окружения: {', '.join(missing_vars)}")

# Конфигурация Gemini
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Neo4j конфигурация
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Инициализация Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    raise Exception(f"Ошибка конфигурации Gemini API: {str(e)}")

def get_gemini_model():
    try:
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash-002",
            generation_config=GENERATION_CONFIG,
            system_instruction="ассистент"
        )
    except Exception as e:
        raise Exception(f"Ошибка создания модели Gemini: {str(e)}")