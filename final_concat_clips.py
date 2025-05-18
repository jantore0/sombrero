from moviepy.editor import VideoFileClip, concatenate_videoclips
import glob
import os

def main():
    # 1. Проверяем существование папки
    if not os.path.exists("highlight_clips"):
        raise FileNotFoundError("Папка highlight_clips не найдена. Сначала запустите make_highlight_chunks.py")
    
    # 2. Получаем список всех highlight-файлов
    clip_files = sorted(glob.glob(os.path.join("highlight_clips", "highlight*.mp4")))
    
    if not clip_files:
        available_files = os.listdir("highlight_clips")
        raise FileNotFoundError(
            f"Не найдено ни одного highlight-файла.\n"
            f"Содержимое папки highlight_clips: {available_files}"
        )
    
    # 3. Загружаем клипы с обработкой ошибок
    clips = []
    for file in clip_files:
        try:
            clip = VideoFileClip(file)
            clips.append(clip)
            print(f"Успешно загружен: {file} ({clip.duration:.2f} сек)")
        except Exception as e:
            print(f"Ошибка загрузки {file}: {str(e)}")
            continue
    
    if not clips:
        raise ValueError("Не удалось загрузить ни одного клипа")
    
    # 4. Объединяем клипы
    final_clip = concatenate_videoclips(clips)
    output_path = "final_highlights.mp4"
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24
    )
    print(f"\nГотово! Создан файл {output_path} из {len(clips)} клипов")

if __name__ == "__main__":
    main()