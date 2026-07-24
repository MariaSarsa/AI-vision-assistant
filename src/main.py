import cv2
from llm_engine import query_llm  
from vision_engine import VisionEngine


# Here in the main we bring together the vision and LLM engines through a loop.


def main():
    print("Initializing AI Vision Assistant...")
    
    # Starting the camera 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    # Loading the vision engine
    vision_sys = VisionEngine()
    
    print("\n Press 'space' to analyze the current frame, 'new' to obtain a new frame, or 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to obtain a frame.")
            break
            
        # Showing the live window
        cv2.imshow("AI Vision Assistant", frame)
        
        key = cv2.waitKey(1) & 0xFF # 0xFF takes only last 8-bit character code to avoid slight differences across systems
        
        # if 'space' is pressed to analyze frame
        if key == ord(' '):
            print("\nFrame captured.")
            
            # Vision engine "sees" the frame
            description = vision_sys.describe_frame(frame)
            
            # Giving the LLM engine the context prompt
            sys_msg = (
                "You are an advanced robotic vision analyzer. "
                "You must deeply analyze the visual context (colors, shapes, elements in the scene, number of similar elements...) provided and answer the user's question. "
            )
            while True:
                user_question = input("\nWhat object/status would you like to verify? (e.g., 'Are there pictures on the scene? How many people are? ...'): ")
                # 'new' to break out and take a new frame
                if user_question.lower() in ['new']:
                    print("\nReturning to live camera feed...")
                    break
                
                # 'q' to quit 
                if user_question.lower() in ['q']:
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                user_msg = f"AI Assistant: I am looking at {description}.\nUser Question: {user_question}"
            
                print("\nAnalyzing answer...")
                response = query_llm(sys_msg, user_msg)
                print(f"\nAI Assistant: {response}\n")
                print("--------------------------------------------------")
                print("Press 'space' to capture a new frame, or 'q' to exit.")
                
        # if 'q' is pressed to exit
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() # Closing everything

if __name__ == "__main__":
    main()