import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class VisionEngine:
    def __init__(self):
        # We use the base model of Florence-2 so that it runs fast on CPU
        self.model_id = 'microsoft/Florence-2-base'

        # Loading processor and NN model weights
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
        
    def describe_frame(self, frame):
        # Convert OpenCV format (BGR) to PIL format (RGB) 
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Ask the model to describe the scene with a complete and comprenhensive description. "pt" stands for Pytorch tensors.
        inputs = self.processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt")

        # Running the image through the vision model to obtain the output text tokens
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024 # We choose a maximum of tokens to avoid extremly long descriptions
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # From tokens to readeable text.

if __name__ == "__main__":
    # Testing: Access the webcam and describe one frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        engine = VisionEngine()
        print("Analyzing frame...")
        description = engine.describe_frame(frame)
        print(f"Scene Description: {description}")
        #cv2.imwrite("test_frame_vision_engine.jpg", frame) # Saving testing frame to check it
    cap.release()