# 🎙️ AI Meeting Assistant

An end-to-end intelligent assistant designed to transform raw meeting audio into actionable insights. Powered by OpenAI Whisper and LangChain, it automatically handles transcription, structured summarization, action-item extraction, and semantic search over past meeting discussions using a modern RAG architecture.

## ✨ Features

- **Audio Transcription**: Highly accurate speech-to-text conversion using OpenAI's Whisper API.
- **Smart Summarization**: Generates clear executive summaries, extracts key decisions, and assigns action items automatically using structured outputs.
- **Interactive Q&A (RAG)**: Chat with your meeting! Uses ChromaDB and LangChain's LCEL (LangChain Expression Language) to retrieve exact details from meeting transcripts accurately.
- **Semantic Search**: Quickly find important information from previous meeting discussions using vector embeddings.
- **Modern UI**: Clean and intuitive web interface built with Streamlit.

---

## 🛠️ Tech Stack

### Backend & AI
- Python
- LangChain (LCEL)
- OpenAI GPT-4o-mini
- OpenAI Whisper-1
- OpenAI text-embedding-3-small

### Vector Database
- ChromaDB

### Frontend
- Streamlit

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-meeting-assistant.git
cd ai-meeting-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## 💡 How to Use

1. Upload a meeting audio file (`.mp3`, `.wav`, or `.m4a`).
2. Click **"Analizə Başla"** to process the audio.
3. The application will:
   - Transcribe the meeting using Whisper
   - Generate a structured summary
   - Extract key decisions
   - Identify action items
   - Store the transcript in ChromaDB for semantic retrieval
4. Ask context-aware questions in the chat panel to retrieve information from the meeting using RAG.

---

## 📂 Project Architecture

```
Audio File
     │
     ▼
OpenAI Whisper
     │
     ▼
Transcript
     │
     ▼
LangChain LCEL Pipeline
     │
     ├── Executive Summary
     ├── Key Decisions
     ├── Action Items
     └── Vector Embeddings
               │
               ▼
            ChromaDB
               │
               ▼
      Retrieval-Augmented Generation (RAG)
               │
               ▼
      Context-Aware Chat Assistant
```

---

## 🎯 Use Cases

- Team meeting summarization
- Project management
- Sprint retrospectives
- Client meeting documentation
- Interview transcription
- Knowledge management
- Action item tracking

---

## 📌 Future Improvements

- Speaker diarization (PyAnnote)
- FastAPI backend
- Authentication
- Multi-user workspace
- Cloud deployment
- Meeting history management
- PDF & DOCX export
- Calendar integration

---

## 📄 License

This project is intended for educational and portfolio purposes.