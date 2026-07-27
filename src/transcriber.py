import os
from openai import OpenAI
from dotenv import load_dotenv

# .env faylındakı API key-i yükləyirik
load_dotenv()

class Transcriber:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe_audio(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio faylı tapılmadı: {file_path}")

        try:
            with open(file_path, "rb") as audio_file:
                transcript_response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript_response
        except Exception as e:
            raise RuntimeError(f"Transkripsiya zamanı xəta baş verdi: {str(e)}")

# Aşağıdakı hissə yalnız bu faylı birbaşa işlətdikdə test etmək üçündür
if __name__ == "__main__":
    transcriber = Transcriber()
    
    # Layihə qovluğuna 'test.mp3' adlı kiçik bir səs faylı atıb yoxlaya bilərsən
    test_audio_path = "test.m4a"    
    
    if os.path.exists(test_audio_path):
        print(f"'{test_audio_path}' faylı oxunur və Whisper API-yə göndərilir...")
        try:
            result = transcriber.transcribe_audio(test_audio_path)
            print("\n--- TRANSKRİPSİYA NƏTİCƏSİ ---")
            print(result)
        except Exception as e:
            print("Xəta:", e)
    else:
        print(f"Test üçün fayl tapılmadı. Zəhmət olmasa layihənin ana qovluğuna (src-nin çölünə) 'test.mp3' adlı səs faylı əlavə edin.")