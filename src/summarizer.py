import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env faylındakı API key-i oxuyuruq
load_dotenv()

# Çıxarılacaq məlumatın strukturunu təyin edirik
class MeetingSummary(BaseModel):
    summary: str = Field(description="Görüşün 3-4 cümləlik qısa xülasəsi")
    decisions: list[str] = Field(description="Görüşdə qəbul edilən əsas qərarlar")
    action_items: list[str] = Field(description="Kimə hansı tapşırığın verildiyi (Məsələn: '@Ad - Tapşırıq')")

class Summarizer:
    def __init__(self):
        # Ucuz, ağıllı və sürətli olan gpt-4o-mini modelini seçirik
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # LLM üçün sistem təlimatı (prompt)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Sən peşəkar bir AI köməkçisisən. Verilən görüş transkriptini analiz edib, strukturlu formada xülasə çıxarmalısan. Dil olaraq transkriptin dilinə uyğun cavab ver."),
            ("human", "Transkript: {transcript}")
        ])
        
        # Zənciri (chain) qururuq və LLM-dən Pydantic modelimizə uyğun cavab istəyirik
        self.chain = self.prompt | self.llm.with_structured_output(MeetingSummary)

    def summarize(self, transcript: str) -> MeetingSummary:
        if not transcript or len(transcript.strip()) == 0:
            raise ValueError("Transkript boş ola bilməz.")
        
        return self.chain.invoke({"transcript": transcript})

# Modulu birbaşa işlətdikdə test etmək üçün hissə
if __name__ == "__main__":
    # Test üçün süni bir görüş mətni
    sample_transcript = """
    Əli: Salam hamıya. Gəlin dərhal başlayaq. Layihənin vaxtını bir az uzada bilərik?
    Aydan: Salam. Bəli, mənə də vaxt lazımdır. Gəlin gələn həftəyə, cümə gününə keçirək deadline-ı.
    Əli: Əla. O zaman belə qərar verdik ki, deadline gələn həftəyə keçirilir. Aydan, sən frontend-i bitirərsən o vaxta qədər?
    Aydan: Bəli, frontend məndədir. Sən də backend API-larını yazarsan, zəhmət olmasa.
    Əli: Oldu, backend API-larını mən həll edəcəm.
    """
    
    print("Xülasə yaradılır (GPT-4o-mini işləyir)...")
    try:
        summarizer = Summarizer()
        result = summarizer.summarize(sample_transcript)
        
        print("\n--- GÖRÜŞÜN XÜLASƏSİ ---")
        print(f"XÜLASƏ:\n{result.summary}\n")
        print(f"QƏRARLAR:\n{result.decisions}\n")
        print(f"TAPŞIRIQLAR:\n{result.action_items}")
    except Exception as e:
        print("Xəta baş verdi:", e)