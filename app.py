import streamlit as st
import torch
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
from pypdf import PdfReader
from groq import Groq
import time

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Генератор Саммари", page_icon="📄", layout="wide")

# --- КЭШИРОВАНИЕ МОДЕЛЕЙ ---
@st.cache_resource
def load_local_model(language="English"):
    if language == "Русский":
        model_name = "cointegrated/rut5-base-multitask"
    else:
        model_name = "facebook/bart-large-cnn"
    
    st.info(f"Загрузка локальной модели: {model_name}...")
    try:
        task = "text2text-generation" if "t5" in model_name else "summarization"
        summarizer = pipeline(task, model=model_name)
        st.success("Локальная модель успешно загружена!")
        return summarizer
    except Exception as e:
        st.error(f"Ошибка при загрузке модели '{model_name}': {e}")
        return None

# --- ФУНКЦИИ ДЛЯ СУММАРИЗАЦИИ ---
def summarize_with_local_model(text_to_summarize, summarizer, min_length=50, max_length=200):
    if not summarizer:
        return "Локальная модель не была загружена."
    summary_list = None # Инициализируем переменную
    try:
        is_t5 = "t5" in summarizer.model.name_or_path
        if is_t5:
            text_to_summarize = "summarize: " + text_to_summarize
        
        summary_list = summarizer(text_to_summarize, max_length=max_length, min_length=min_length, do_sample=False)
        
        if is_t5:
            return summary_list[0]['generated_text']
        else:
            return summary_list[0]['summary_text']
            
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        if summary_list:
            print(f"Вывод модели, который вызвал ошибку: {summary_list}")
        return f"Ошибка при генерации саммари с помощью локальной модели: {e}"

def summarize_with_groq(text_to_summarize, groq_client, model_name, language):
    if language == "Русский":
        prompt = f"""
        Твоя задача - сделать краткую, но содержательную выжимку (саммари) предоставленного текста на русском языке.
        Саммари должно отражать ключевые идеи, факты и выводы. Сохраняй нейтральный и информативный тон.
        Текст для обработки:
        ---
        {text_to_summarize}
        ---
        Краткая выжимка (саммари):
        """
    else: # English prompt
        prompt = f"""
        Your task is to create a concise yet comprehensive summary of the provided text in English.
        The summary should reflect the key ideas, facts, and conclusions. Maintain a neutral and informative tone.
        Text to process:
        ---
        {text_to_summarize}
        ---
        Concise summary:
        """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model_name)
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Ошибка при обращении к Groq API: {e}"

def map_reduce_summarization(text, model_choice, language, api_key=None, local_summarizer=None):
    if model_choice == 'Mistral (Groq API)':
        if not api_key:
            st.error("Пожалуйста, введите ваш Groq API ключ.")
            return None
        chunk_size, chunk_overlap = 8000, 400
        client = Groq(api_key=api_key)
        model_name = "llama3-70b-8192"
        summarize_func = lambda chunk: summarize_with_groq(chunk, client, model_name, language)
    else:
        if not local_summarizer:
            st.error("Локальная модель не загружена.")
            return None
        chunk_size, chunk_overlap = 2000, 200
        summarize_func = lambda chunk: summarize_with_local_model(chunk, local_summarizer)

    if len(text) < chunk_size * 1.2:
        st.info("Текст короткий, генерируем саммари напрямую...")
        return summarize_func(text)
    st.info("Текст длинный, применяется стратегия Map-Reduce...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = text_splitter.split_text(text)
    st.write(f"Текст разбит на {len(chunks)} частей. Генерация промежуточных саммари...")
    initial_summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = summarize_func(chunk)
        initial_summaries.append(summary)
        time.sleep(0.1)
        progress_bar.progress((i + 1) / len(chunks))
    st.write("Объединение результатов и генерация финального саммари...")
    combined_summary = "\n\n".join(initial_summaries)
    if len(combined_summary) > chunk_size * 1.2:
        st.warning("Промежуточные саммари все еще слишком длинные. Повторное сжатие...")
        final_summary = map_reduce_summarization(combined_summary, model_choice, language, api_key, local_summarizer)
    else:
        final_summary = summarize_func(combined_summary)
    return final_summary

# --- ФУНКЦИИ ДЛЯ ЧТЕНИЯ ФАЙЛОВ ---
def read_text_file(uploaded_file): return uploaded_file.read().decode("utf-8")
def read_docx_file(uploaded_file): return "\n".join([para.text for para in Document(uploaded_file).paragraphs])
def read_pdf_file(uploaded_file): return "\n".join([page.extract_text() for page in PdfReader(uploaded_file).pages])

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("📄 Генератор Кратких Саммари")
st.markdown("Вставьте текст или загрузите файл, чтобы получить его краткую выжимку.")
with st.sidebar:
    st.header("⚙️ Настройки")
    language = st.selectbox("1. Выберите язык текста:", ("Русский", "English"))
    model_choice = st.radio("2. Выберите модель:", ('Локальная модель', 'Mistral (Groq API)'))
    api_key = None
    if model_choice == 'Mistral (Groq API)':
        api_key = st.text_input("Введите ваш Groq API ключ:", type="password")
    st.markdown("---")
    st.info("Длинные тексты обрабатываются методом Map-Reduce.")

input_method = st.radio("Выберите способ ввода:", ("Вставить текст", "Загрузить файл"), horizontal=True)
user_text = ""
if input_method == "Вставить текст":
    user_text = st.text_area("Введите ваш текст здесь:", height=300)
else:
    uploaded_file = st.file_uploader("Выберите файл (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    if uploaded_file:
        with st.spinner("Чтение файла..."):
            if uploaded_file.name.endswith(".txt"): user_text = read_text_file(uploaded_file)
            elif uploaded_file.name.endswith(".docx"): user_text = read_docx_file(uploaded_file)
            elif uploaded_file.name.endswith(".pdf"): user_text = read_pdf_file(uploaded_file)
        st.success("Файл успешно загружен!")

if st.button("✨ Сгенерировать саммари", type="primary"):
    if not user_text.strip():
        st.warning("Пожалуйста, введите текст или загрузите файл.")
    else:
        local_summarizer = None
        if model_choice == 'Локальная модель':
            with st.spinner(f"Загрузка локальной модели для языка '{language}'... (может занять время)"):
                local_summarizer = load_local_model(language)
        
        with st.spinner("Магия в процессе... Генерируем саммари..."):
            final_summary = map_reduce_summarization(
                text=user_text,
                model_choice=model_choice,
                language=language,
                api_key=api_key,
                local_summarizer=local_summarizer
            )
        if final_summary:
            st.subheader("🎉 Ваше саммари готово:")
            st.markdown(f"> {final_summary}")