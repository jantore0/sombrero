import librosa
import numpy as np
import pandas as pd
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import matplotlib.pyplot as plt

# 1. Подготовка папки для клипов
os.makedirs("highlight_clips", exist_ok=True)

# 2. Загрузка аудио
filename = "./audio.wav"
try:
    x, sr = librosa.load(filename, sr=16000)
    
    # 3. Исправленный вызов get_duration
    duration = librosa.get_duration(y=x, sr=sr)  # Новый синтаксис
    print(f"Аудио загружено. Длительность: {duration:.2f} сек ({duration/60:.2f} мин)")
    
    # 4. Анализ энергии
    max_slice = 10  # Длина сегмента в секундах
    window_length = max_slice * sr
    
    # 5. Расчет энергии для каждого сегмента
    energy = np.array([
        sum(abs(x[i:i+window_length]**2)) 
        for i in range(0, len(x), window_length)
    ])
    
    # 6. Настройка порога (можно регулировать)
    thresh = np.percentile(energy, 85)  # Автоматический порог (топ 15% по энергии)
    print(f"Автоматический порог энергии: {thresh:.2f}")
    
    # 7. Поиск интересных моментов
    highlights = []
    for i, e in enumerate(energy):
        if e >= thresh:
            start = i * max_slice
            end = (i + 1) * max_slice
            highlights.append((e, start, end))
    
    # 8. Создание клипов
    for i, (e, start, end) in enumerate(highlights):
        output_file = f"highlight_clips/highlight_{i+1}.mp4"
        print(f"Создаю клип {i+1}: {start:.1f}-{end:.1f} сек (энергия: {e:.2f})")
        
        try:
            ffmpeg_extract_subclip(
                "videofile.mp4",
                max(0, start - 2),  # Начинаем на 2 сек раньше
                min(duration, end + 2),  # Заканчиваем на 2 сек позже
                targetname=output_file
            )
        except Exception as e:
            print(f"Ошибка при создании клипа: {str(e)}")

except Exception as e:
    print(f"Критическая ошибка: {str(e)}")