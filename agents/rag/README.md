# Arxiv RAG

## Prerequisite
```pip install streamlit langchain```

## LLM Serving
1. Use OpenAI API service
    Obtain an OpenAI API key and set the OpenAI base URL (Refer to https://openai.com/api/)
2. Deploy LLM locally with OpenAI API serving
    * Use FastChat to serve local LLM
        ```
        python3 -m fastchat.serve.controller
        python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
        python3 -m fastchat.serve.openai_api_server --host localhost --port 8001
        ```
## Get Started
1. Copy `.env.sample` to `.env` and set env variables
2. Run demo
    ```
    python3 chain.py
    ```
3. Run Streamlit App
    ```
    streamlit run app.py
    ```
