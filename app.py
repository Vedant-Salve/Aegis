import os
import cv2
import shutil
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' hides INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disables the specific oneDNN warning

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --- Import your Aegis modules ---
from Aegis.cloaking import AegisCloakingEngine
from Aegis.utils import show_comparison, save_image
from Aegis.FaceRedactor import FaceRedactor
from Aegis.TextRedactor import TextRedactor

# Global Configuration
TARGET_POOL = "./targets/"
TEMP_FOLDER = "./temp_aegis_pipeline/"

def preview_and_ask_save(input_path, result):
    """Shows the final comparison and prompts the user to save."""
    if result is None:
        print("⚠️ No final image to display or save.")
        return

    print("\nDisplaying final comparison...")
    # This will trigger your matplotlib side-by-side plot
    show_comparison(input_path, result)
    
    choice = input("\nDo you want to save the final image? (y/n): ").strip().lower()
    if choice == 'y':
        folder = input("Enter destination folder (Press Enter for default 'assets/generatedImages/'): ").strip()
        
        if folder:
            saved_path = save_image(result, folder_path=folder)
        else:
            # Relies on the default folder_path in your utils
            saved_path = save_image(result) 
            
        if saved_path:
            print(f"✅ Successfully saved final output to: {saved_path}")
    else:
        print("❌ Image discarded.")

def get_image_path_from_user():
    """Helper to get a valid image path from the user."""
    while True:
        path = input("\n[INPUT] Enter the path to your input image: ").strip().strip('"').strip("'")
        if os.path.exists(path) and os.path.isfile(path):
            return path
        print("❌ Error: File not found. Please try again.")

def task_redact_faces(input_path):
    """Redacting Faces"""
    print("\n--- FACE REDACTION ENGINE ---")    
    redactor = FaceRedactor()
    
    # Load and Detect
    img = redactor.load_image(input_path)
    boxes = redactor.get_detections(img)
    
    if len(boxes) == 0:
        print("⚠️ No faces detected in this image.")
        return img # Return unmodified image to keep pipeline moving

    # Show Preview 
    print("Displaying preview with box numbers...")
    redactor.plot_preview(img, boxes)
    
    # Get User Input
    box_input = input("Enter box numbers to blur (e.g., '1 2'): ")
    
    # Parse Input
    try:
        to_blur = [int(x) for x in box_input.replace(',', ' ').split()]
    except ValueError:
        print("❌ Invalid input. Skipping redaction.")
        return img

    # Apply and Save
    result = redactor.apply_redaction(img, boxes, to_blur)
    return result

def task_redact_pii(input_path):
    """Redacting PII (Text)"""
    print("\n--- PII REDACTION ENGINE ---")
    
    t_redactor = TextRedactor()
    
    print("Scanning image for text...")
    text_detections = t_redactor.detect_text(input_path)
    img = cv2.imread(input_path)
    
    if not text_detections:
        print("⚠️ No text detected.")
        return img # Return unmodified image to keep pipeline moving
        
    print(f"✅ Detected {len(text_detections)} text regions.")
    
    # Apply Auto Redaction
    result = t_redactor.auto_redact(img, text_detections)
    return result

def task_cloaking(input_path):
    """Cloaking Image"""
    print("\n--- AI CLOAKING ENGINE ---")
    
    print("Initializing Aegis Cloaking Engine (Loading Models)...")
    cloaker = AegisCloakingEngine(target_pool_path=TARGET_POOL)
    
    print("Generating Cloak... (This may take a moment)")
    cloaked_face = cloaker.generate_cloak(
        input_path=input_path,
        iterations=1100,        
        lr=0.02,               
        rho=0.007,             
        penalty_lambda=15000.0 
    )
    return cloaked_face

def main():
    """Main Application Loop"""
    while True:
        print("\n" + "="*40)
        print("            🛡️    AEGIS   🛡️")
        print("="*40)
        print("1. Redact Faces")
        print("2. Redact PII")
        print("3. Cloaking")
        print("4. Quit")
        print("-" * 40)
        
        choice = input("\nEnter tasks to perform (e.g., '1 2 3' or '2 1') or '4' to quit: ").strip()
        
        if choice == '4':
            print("\nExiting Aegis...")
            break
            
        # Parse the input into a sorted list of unique integers
        try:
            tasks = sorted(list(set([int(x) for x in choice.replace(',', ' ').split()])))
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by spaces.")
            continue
            
        if not all(t in [1, 2, 3] for t in tasks):
            print("❌ Invalid task selected. Please only use 1, 2, or 3.")
            continue
            
        input_path = get_image_path_from_user()
        current_path = input_path
        final_image = None
        
        # --- PIPELINE EXECUTION ---
        for task in tasks:
            if task == 1:
                final_image = task_redact_faces(current_path)
            elif task == 2:
                final_image = task_redact_pii(current_path)
            elif task == 3:
                final_image = task_cloaking(current_path)
                
            if final_image is None:
                print(f"⚠️ Task {task} failed and returned None. Halting pipeline.")
                break
                
            # Use your save_image function to create a temporary intermediate file
            # It will auto-generate a name inside TEMP_FOLDER and return the path
            temp_saved_path = save_image(final_image, folder_path=TEMP_FOLDER)
            
            if temp_saved_path:
                current_path = temp_saved_path
            else:
                print("❌ Failed to save intermediate pipeline step. Halting.")
                final_image = None
                break

        # --- FINAL PREVIEW & SAVE ---
        if final_image is not None:
            preview_and_ask_save(input_path, final_image)
            
        # Clean up the intermediate pipeline files
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER, ignore_errors=True)

if __name__ == "__main__":
    main()