# AI-VISION-ASSISTANT

A simple project made to merge computer vision mechanics with local AI, to learn how to create a camera assistant for spatial reasoning.

By combining **Microsoft Florence-2** for visual perception and **Microsoft Phi-3** for natural language reasoning, this system answers questions about scenes from the computer webcam in real time.

* **Perception Engine:** [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base) handles visual analysis, object description, and scene parsing.
* **Reasoning Engine:** [Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) manages concise natural language question-answering based on captured visual contexts.

It is 100% local execution, with separated perception and reasoning engines, so that it is CPU friendly.
