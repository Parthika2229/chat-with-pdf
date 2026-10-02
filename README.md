# Document Q&A App

A Streamlit app that lets you upload a document and ask questions about it. An AI model reads the document and answers based on its content.

**Live app:** https://document-app-uvneh7pt35.streamlit.app/

## Project Structure

```
├── streamlit_app.py     # Main Streamlit app (UI and AI logic)
├── requirements.txt     # Python dependencies
├── LICENSE              # Apache-2.0 license
└── README.md            # Project documentation
```

## How It Works

1. **Upload:** The user uploads a text or markdown file (`.txt` or `.md`).
2. **Ask:** The user types a question about the document.
3. **Read:** The app reads the file and combines the document text with the question into one prompt.
4. **Answer:** The prompt is sent to the OpenAI API, and the model generates an answer based only on the document.
5. **Display:** The answer is shown in the app.

```
Upload file → Enter question → Document + question sent to OpenAI → Answer displayed
```

## Tech Stack

- Python
- Streamlit
- OpenAI API
