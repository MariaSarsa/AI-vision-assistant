# AI-VISION-ASSISTANT

A simple project made to merge computer vision mechanics with local AI, to learn how to create a camera assistant for spatial reasoning.

By combining **Microsoft Florence-2** for visual perception and **Microsoft Phi-3** for natural language reasoning, this system answers questions about scenes from the computer webcam in real time.

* **Perception Engine:** [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base) handles visual analysis, object description, and scene parsing.
* **Reasoning Engine:** [Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) manages concise natural language question-answering based on captured visual contexts.

It is 100% local execution, with separated perception and reasoning engines, so that it is CPU friendly.


## Repository Structure

The workspace is laid out as follows:

```text
├── src/
│   ├── vision_engine.py    # Florence-2 perception model
│   ├── llm_engine.py       # Phi-3 reasoning engine
│   └── main.py             # Complete pipeline
├── .gitignore              
├── requirements.txt        # Python environment dependencies
└── README.md               # Project documentation
```

## Quickstart

1. **Create virtual environment
```bash
python -m venv venv
```

2. **Activate virtual environment
   ***On Windows:
  ```bash
  venv\Scripts\activate
```
  ***On macOS/Linux:
```bash
  source venv/bin/activate
```

3. **Install dependencies
```bash
pip install -r requirements.txt
```

4. **Activate your webcam

5. **Launch Assistant
python src/main.py
```
