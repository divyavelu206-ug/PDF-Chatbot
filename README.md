# PDF Chatbot – Retrieval Augmented Generation (RAG) using FastAPI

A beginner-friendly, high-performance AI-powered PDF Chatbot built with **FastAPI**, **PyPDF**, **LangChain Text Splitters**, **Sentence Transformers**, **FAISS**, and **Groq Cloud API**.

Users can upload any PDF document and ask questions about its content. The system uses a local vector database (FAISS) to retrieve the most relevant text chunks and prompts a Groq-hosted Large Language Model (e.g. LLaMA 3.3 70B) to generate accurate answers based **only** on the uploaded document.

---

## 1. Project Folder Structure

```
rag-pdf-chatbot/
│
├── app/
│   ├── main.py              # FastAPI endpoints, static file mounting & app initialization
│   ├── pdf_processor.py     # PDF text extraction (PyPDF) & chunking (LangChain)
│   ├── embedding.py         # Local Sentence Transformers embedding model manager
│   ├── vector_store.py      # FAISS vector database index and top-k similarity search
│   ├── rag_pipeline.py      # RAG coordinator connecting FAISS context with Groq LLM
│   │
│   ├── templates/
│   │   └── index.html       # Clean and responsive HTML5 frontend
│   │
│   └── static/
│       ├── style.css        # Modern white and light-blue responsive styling
│       └── script.js        # Drag-and-drop, asynchronous API calls, and DOM manipulation
│
├── uploads/                 # Local directory where uploaded PDF files are saved
│   └── .gitkeep
├── requirements.txt         # Project dependencies
├── .env                     # Local environment file (contains GROQ_API_KEY)
├── .env.example             # Example environment template
├── .gitignore               # Git ignore rules for venv, .env, and uploaded files
└── README.md                # Complete setup and usage documentation
```

---

## 2. Technology Stack

- **Backend:** Python 3.10+
- **API Framework:** FastAPI & Uvicorn
- **PDF Extraction:** PyPDF
- **Chunking:** LangChain RecursiveCharacterTextSplitter (`chunk_size=500`, `chunk_overlap=50`)
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2` – runs completely locally on CPU)
- **Vector Database:** FAISS (`faiss-cpu`)
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla ES6)

---

## 3. RAG Architecture & Workflow

```
[User uploads PDF]
        │
        ▼
[PyPDF extracts raw text from PDF pages]
        │
        ▼
[LangChain splits text into chunks: size=500, overlap=50]
        │
        ▼
[Sentence Transformers generates 384-dimensional dense vectors]
        │
        ▼
[Vectors & Text Chunks stored in FAISS in-memory index]
        │
[User asks a question]
        │
        ▼
[Question converted to 384-dimensional query vector]
        │
        ▼
[FAISS retrieves top-3 most similar text chunks]
        │
        ▼
[Context + Question sent to Groq LLM with strict instructions]
        │
        ▼
[Groq generates answer or returns fallback message]
        │
        ▼
[Frontend displays answer along with top-3 source passages]
```

---

## 4. Setup & Installation (Windows & VS Code)

Follow these step-by-step instructions in PowerShell or Windows Command Prompt.

### Step 1: Open the Project in VS Code
Open VS Code, press `Ctrl + \`` to open the integrated terminal, and ensure you are in the project folder:
```powershell
cd rag-pdf-chatbot
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment ensures clean dependency isolation:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
On Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` and try again).*

On Windows Command Prompt (cmd):
```cmd
venv\Scripts\activate.bat
```

You will see `(venv)` appear at the beginning of your command line.

### Step 4: Install Dependencies
Upgrade pip and install all required libraries:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Your Groq API Key
1. Visit [Groq Cloud Console](https://console.groq.com/keys) and create a free account.
2. Click **Create API Key**, copy the generated key.
3. Open the `.env` file in the `rag-pdf-chatbot/` directory and replace the placeholder:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

---

## 5. Running the Application

Start the FastAPI application with Uvicorn:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

When you see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
The server is ready!

> **Note on First Run:** On the very first upload or query, Sentence Transformers will automatically download the lightweight `all-MiniLM-L6-v2` embedding model (~80 MB) into your local cache. Subsequent runs load instantly from local disk.

---

## 6. Accessing the Frontend

Open your web browser and navigate to:
```
http://localhost:8000
```
or
```
http://127.0.0.1:8000
```

### Using the Interface:
1. **Upload a PDF:** Drag and drop your `.pdf` file or click to select one, then click **Upload & Process PDF**.
2. **Review Upload Status:** A green confirmation alert will display the filename and the number of chunks indexed in FAISS.
3. **Ask Questions:** Type any question regarding the PDF in the input box and click **Ask**.
4. **View Response:** The AI generates an answer strictly from the document. The top 3 retrieved context passages and their L2 similarity scores will be displayed under **Source Information**.
5. **Fallback:** If you ask a question about something not in the document, the chatbot responds:
   > *"Sorry, the answer is not available in the uploaded document."*

---

## 7. Interactive Swagger API Testing

FastAPI provides an automatic, interactive API documentation interface.

Open your browser to:
```
http://localhost:8000/docs
```

You can test every endpoint directly:

1. **`GET /health`**
   - Click **Try it out** → **Execute**.
   - Check if backend is active and inspect the currently loaded document.

2. **`POST /upload`**
   - Click **Try it out**.
   - Click **Choose File** to pick a `.pdf` file from your computer.
   - Click **Execute** to verify the response containing `status`, `filename`, `total_chunks`, and `character_count`.

3. **`POST /ask`**
   - Click **Try it out**.
   - Provide a JSON request body:
     ```json
     {
       "question": "What is the main topic of the document?"
     }
     ```
   - Click **Execute** to view the generated answer and retrieved source chunks.

---

## 8. Troubleshooting Common Errors

### 1. `GROQ_API_KEY is missing or invalid`
- **Cause:** `.env` does not exist or still contains `your_groq_api_key_here`.
- **Solution:** Open `.env` and paste your actual key starting with `gsk_`. Save the file and restart the server.

### 2. `Execution of scripts is disabled on this system` (PowerShell)
- **Cause:** Windows PowerShell execution policy restricts script execution.
- **Solution:** Run this command in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  Then reactivate: `.\venv\Scripts\Activate.ps1`.

### 3. `No extractable text found in the PDF`
- **Cause:** The PDF consists solely of scanned photos/images without embedded text layers.
- **Solution:** Standard PyPDF extracts embedded digital text. Test with a standard text-based PDF (such as research papers, e-books, or exported Word documents).

### 4. `faiss-cpu installation error` on Windows
- **Cause:** Some older Python versions on Windows might have wheel compatibility issues.
- **Solution:** Ensure you are using Python 3.10, 3.11, or 3.12 (64-bit). Run `pip install faiss-cpu` directly.

---

## 9. License & Credits

Built as a clean reference implementation of Retrieval-Augmented Generation (RAG) for Python developers. Feel free to customize and expand!
