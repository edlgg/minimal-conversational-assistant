# Minimal Conversational Assistant Framework & UI

## Description

Bootstrap template for chatbot UI with llm based conversational assistants using:

- Lang Chain
- FastAPI
- HTMX
- DaisyUI
- Plotly

## Set Up

```python
python3.10 -m venv .venv/   
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```python
source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
python src/ui/main.py
```

## References

- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [LangChain LLM Chain Tutorial](https://python.langchain.com/docs/tutorials/llm_chain/)
- [LangChain Agents Tutorial](https://python.langchain.com/docs/tutorials/agents/)
- [LangChain Concepts](https://python.langchain.com/docs/concepts/)
- [LangChain Custom Tools](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/#subclass-basetool)
- [ARXIV Paper](https://arxiv.org/pdf/2411.05285)
