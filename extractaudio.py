import moviepy.editor as mp
import os

input_video = "./videofile.mp4"
output_audio = "./audio.wav"

try:
    # Проверка исходного файла
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Видеофайл {input_video} не найден!")

    # Извлечение аудио
    clip = mp.VideoFileClip(input_video)
    clip.audio.write_audiofile(output_audio)
    
    # Проверка результата
    if os.path.exists(output_audio):
        print(f"Аудио успешно извлечено в {output_audio}")
        print(f"Длительность: {clip.duration:.2f} секунд")
    else:
        raise Exception("Не удалось создать аудиофайл")

except Exception as e:
    print(f"Ошибка: {str(e)}")