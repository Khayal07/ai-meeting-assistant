import streamlit as st
import os
import tempfile
from src.transcriber import Transcriber
from src.summarizer import Summarizer
from src.rag_engine import RAGEngine

# --- 1. SƏHİFƏNİN TƏNZİMLƏMƏLƏRİ ---
st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️", layout="wide")

# --- 2. YADDAŞ (Session State) İNİSİALİZASİYASI ---
# Streamlit hər dəfə yenilənəndə məlumatlar itməsin deyə yaddaşda saxlayırıq
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🎙️ AI Meeting Assistant")
st.write("Səs faylını yüklə, mətnə çevir, xülasəsini al və görüşlə bağlı suallar ver!")

# Ekranı iki sütuna bölürük: Sol (Yükləmə) və Sağ (Chat)
col1, col2 = st.columns([1, 1])

# --- SOL PANEL: YÜKLƏMƏ VƏ XÜLASƏ ---
with col1:
    st.header("1. Səs Yükləmə və Xülasə")
    uploaded_file = st.file_uploader("Görüşün səs faylını seçin (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    
    if uploaded_file is not None:
        if st.button("Analizə Başla"):
            with st.spinner("Səs mətnə çevrilir (Whisper işləyir)..."):
                # Faylı müvəqqəti diskdə saxlayırıq ki, Whisper-ə yolunu (path) verə bilək
                with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # 1. Transkripsiya
                    transcriber = Transcriber()
                    transcript_text = transcriber.transcribe_audio(tmp_path)
                    st.session_state.transcript = transcript_text
                    
                    # 2. Xülasə
                    st.info("Xülasə və tapşırıqlar çıxarılır...")
                    summarizer = Summarizer()
                    summary_result = summarizer.summarize(transcript_text)
                    st.session_state.summary = summary_result
                    
                    # 3. RAG Mühərrikinə əlavə etmək
                    st.info("Məlumat vektor bazasına (ChromaDB) yazılır...")
                    st.session_state.rag_engine.add_transcript(transcript_text)
                    
                    st.success("Analiz uğurla bitdi!")
                except Exception as e:
                    st.error(f"Xəta baş verdi: {e}")
                finally:
                    # Müvəqqəti faylı silirik
                    os.remove(tmp_path)
    
    # Xülasəni ekranda göstərmək
    if st.session_state.summary:
        st.subheader("📋 Görüşün Xülasəsi")
        st.write(st.session_state.summary.summary)
        
        st.subheader("✅ Qərarlar")
        for dec in st.session_state.summary.decisions:
            st.markdown(f"- {dec}")
            
        st.subheader("📌 Tapşırıqlar")
        for task in st.session_state.summary.action_items:
            st.markdown(f"- {task}")

# --- SAĞ PANEL: RAG / CHAT ---
with col2:
    st.header("2. Görüşlə Sual-Cavab (Chat)")
    if not st.session_state.transcript:
        st.warning("Chat etmək üçün əvvəlcə sol tərəfdən səs faylı yükləyib analiz edin.")
    else:
        # Əvvəlki mesajları ekrana çıxarırıq
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # İstifadəçi sual yazır
        user_input = st.chat_input("Görüş barədə sualınızı yazın...")
        
        if user_input:
            # İstifadəçinin sualını tarixçəyə əlavə edib göstəririk
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # RAG mühərrikindən cavabı alırıq
            with st.chat_message("assistant"):
                with st.spinner("Görüş daxilində axtarılır..."):
                    response = st.session_state.rag_engine.ask(user_input)
                    st.markdown(response)
            
            # Cavabı tarixçəyə əlavə edirik
            st.session_state.chat_history.append({"role": "assistant", "content": response})