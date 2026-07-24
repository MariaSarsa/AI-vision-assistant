import os
from llama_cpp import Llama

# Locating model in directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "phi3-mini.gguf"))

# Model downloading link 
URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

# If model not downloaded, then  create folder for it
if not os.path.exists(MODEL_PATH):

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("\nDownload complete.")

# Loading the model into the system's RAM
print(f"Loading LLM Engine.")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("LLM Engine Loaded.")

def query_llm(system_prompt: str, user_prompt: str) -> str:
    # Sends prompt to LLM and returns text answer
    try:
        # Here we give the instructions, which is the role of the LLM and the role of the user.
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=128,
            temperature=0.2 # Temperature means how creative we let the model be, low temperature is more deterministic.
        )
        return output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Execution Error: {e}"

if __name__ == "__main__":
    sys_msg = "You are a helpful robotic assistant. Be concise."
    user_msg = "Explain what a LLM is in one sentence."
    
    print("\nRunning test ...")
    response = query_llm(sys_msg, user_msg)
    print(f"\nAnswer:\n{response}")