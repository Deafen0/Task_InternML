from transformers import pipeline
print("Попытка загрузить русскую модель...")
try:
    summarizer = pipeline("summarization", model="cointegrated/rut5-base-multitask")
    print("Успех! Русская модель загружена.")
except Exception as e:
    print("\nОшибка при загрузке русской модели:")
    print(e)

print("-" * 20) #delimeter

print("Попытка загрузить английскую модель...")
try:
    summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Успех! Английская модель загружена.")
except Exception as e:
    print("\nОшибка при загрузке английской модели:")
    print(e)