# Law-Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot designed to assist users with legal inquiries. The chatbot leverages advanced natural language processing techniques to provide accurate and contextually relevant responses based on a curated legal knowledge base.

## TODO

- Create a nice interface using Streamlit or Gradio or Chainlit or Dash or similar framework.

## Reindexing the Knowledge Base

To reindex the knowledge base, run the following command:

```bash
python3 -c 'from vector_db_manager import clear_db; clear_db()'
```