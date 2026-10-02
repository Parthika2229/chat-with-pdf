# Document Q&A App

Upload a document and ask questions about it. The app reads your file and answers using an AI language model, so you don't have to search through the text yourself.

**Live app:** [add your Streamlit link here]

## What it does

- Upload a text or markdown document (`.txt`, `.md`)
- Type a question about the document
- Get an answer based on the document's content

## Why I built it

[Write 1-2 sentences in your own words. For example: "I wanted a quick way to pull answers out of long notes and articles without reading everything."]

## Built with

- Python
- [Streamlit](https://streamlit.io) for the web interface
- [OpenAI API](https://platform.openai.com) for answering questions

## How to run it locally

1. Clone the repository
```bash
   git clone https://github.com/Parthika2229/[your-repo-name].git
   cd [your-repo-name]
```

2. Install the dependencies
```bash
   pip install -r requirements.txt
```

3. Run the app
```bash
   streamlit run streamlit_app.py
```

4. Open the link shown in your terminal (usually `http://localhost:8501`) and enter your API key when asked.

## API key

The app needs an OpenAI API key. You can get one from the [OpenAI platform](https://platform.openai.com/api-keys).

- **Locally:** paste it into the app's sidebar, or add it to `.streamlit/secrets.toml`
- **On Streamlit Community Cloud:** add it under *Advanced settings → Secrets*

Never commit your API key to GitHub.

## Project structure

```
├── streamlit_app.py     # Main app
├── requirements.txt     # Python dependencies
└── README.md
```

## What I changed from the template

[List anything you customized, such as the layout, the prompt, supported file types, or the model. Delete this section if you haven't changed anything yet.]

## Future ideas

- Support PDF and Word files
- Remember previous questions in a chat
- [Add your own]

## Credits and license

Started from Streamlit's [document-qa-template](https://github.com/streamlit/document-qa-template). Licensed under the Apache License 2.0.

---

Built by [Parthika Battala](https://github.com/Parthika2229)
