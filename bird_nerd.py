# python3.13 -m venv venv
# python3 bird-nerd.py

# Start the server: ollama serve.
# Pull a vision model: ollama pull llava:7b

"""
test_pictures/bird_A.jpg
test_pictures/bird_B.jpg
test_pictures/bird_C.jpg
test_pictures/bird_D.jpg
test_pictures/bird_E.jpg
test_pictures/animal_A.jpg

bird_A: American Robin: Turdus migratorius
bird_B: Redwinged Blackbird: Agelaius phoeniceus
bird_C: Northern Cardinal (male): Cardinalis cardinalis
bird_D: Northern Cardinal (female): Cardinalis cardinalis
bird_E: Green Cheeked Conure: Pyrrhura molinae
animal_A: Squirrel: Sciuridae
"""

"""
TODO:
- 1. Look into using faster model
    - 1a. Maybe we can use a model from a different bird ID dataset from GitHub? (Need to give credits)
    - 1b. Each search takes about 30 seconds with llava:7b
    - 1c. https://ollama.com/blog/vision-models
- 2. Add confidence scoring for predictions
"""

import base64
import os
import argparse
from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model = "llava:7b"

class BirdIdentification(BaseModel):
    common_name: str
    scientific_name: str

class IdentificationResult(BaseModel):
    tool_name: Literal["identify_bird"] = "identify_bird"
    thinking: str
    identification: BirdIdentification
    alternative_species: list[str]

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for API."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError("Image file not found: " + image_path)
    except Exception as e:
        raise Exception("Error reading image file: " + str(e))

def identify_bird_with_ollama(image_path: str) -> dict:
    """Identify bird species using Ollama vision model."""
    try:
        base64_image = encode_image_to_base64(image_path) # Encode image
    except Exception as e:
        return {"error": str(e)}
    
    prompt = """Look at this bird image and identify the species. Picture was taken from a backyard in Midwest Michigan.

    Respond with ONLY:
    Common name: [bird's common name]
    Scientific name: [bird's scientific name]

    Be brief and direct."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,  # Smaller token limit
            temperature=0.1  # Lower temperature for speed
        )
        
        response_text = response.choices[0].message.content.strip()
        return {"response": response_text}
            
    except Exception as e:
        return {"error": f"API call failed: {e}"}

def format_identification_result(result: dict, image_path: str):
    """Format and display the bird identification results."""
    print(f"\n{'='*60}")
    print(f"BIRD IDENTIFICATION RESULTS")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if "response" in result:
        print("BIRD IDENTIFICATION:")
        print(result["response"])
        return
    
    print("No identification result found")
    print(f"{'='*60}")

def validate_image_file(image_path: str) -> bool:
    """Check if the image file exists and is a valid format."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = os.path.splitext(image_path.lower())[1]
    
    if file_ext not in valid_extensions:
        print(f"Error: Unsupported image format: {file_ext}")
        print(f"   Supported formats: {', '.join(valid_extensions)}")
        return False
    
    return True



def main():
    parser = argparse.ArgumentParser(description="Identify bird species from images using Ollama AI")
    parser.add_argument("image_path", nargs="?", help="Path to the bird image file")
    parser.add_argument("--model", default="llava:7b", help="Ollama vision model to use (default: llava:7b)")
    
    args = parser.parse_args()
    
    global model
    model = args.model

    print("Welcome to Bird Nerd!")
    # print(f"Using model: {model}")
    # print("This bot uses Ollama AI to identify bird species from your photos.\n")
    
    if args.image_path:
        if validate_image_file(args.image_path):
            print("Analyzing image: " + args.image_path)
            print("Please wait, this may take a moment...")
            
            result = identify_bird_with_ollama(args.image_path)
            format_identification_result(result, args.image_path)
    else:
        while True:
            print("-" * 40)
            image_path = input("Enter the path to your bird image (or 'exit' to quit): ").strip()
            
            if image_path.lower() == 'exit':
                print("Thanks for using Bird Nerd! Goodbye!")
                break
            
            if not image_path:
                print("Please enter a valid image path.")
                continue
            
            image_path = image_path.strip('"\'') # Remove quotes if path has them
            
            if not validate_image_file(image_path):
                continue

            print(f"\nAnalyzing image: {os.path.basename(image_path)}")
            print("Please wait, this may take a moment...")

            result = identify_bird_with_ollama(image_path)
            format_identification_result(result, image_path)
            
            continue_choice = input("\nWould you like to identify another bird? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("Thanks for using Bird Nerd. Goodbye!")
                break

if __name__ == "__main__":
    main()