import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load a lightweight local model (CPU compatible)
print("🔁 Loading model...")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
print("✅ Model loaded!")


def generate_recipe(ingredients: str, cuisine: str = "Any"):
    prompt = f"""
I have the following ingredients: {ingredients}.
Create a {cuisine} style recipe. Include:
- A creative dish name
- Estimated cooking time
- Step-by-step instructions
- Respond in clear and structured format.
"""
    output = generator(prompt, max_new_tokens=300, temperature=0.6, do_sample=True)
    return output[0]["generated_text"]


def listen_microphone(prompt="Speak now..."):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print(prompt)
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("🎧 You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return ""


if __name__ == "__main__":
    print("🍽️  Welcome to the Local AI Recipe Generator!")

    mode = input("Choose input mode (1 = Text, 2 = Voice): ")

    if mode.strip() == "2":
        ingredients = listen_microphone("🎤 Say the ingredients you have (comma-separated):")
        cuisine = listen_microphone("🎤 Say your preferred cuisine or say 'Any':")
        if not cuisine.strip():
            cuisine = "Any"
    else:
        ingredients = input("Enter ingredients (comma-separated): ")
        cuisine = input("Enter preferred cuisine (e.g., Bangladeshi, Italian) or leave empty: ")
        if not cuisine.strip():
            cuisine = "Any"

    print("\n👨‍🍳 Generating your personalized recipe...\n")
    recipe = generate_recipe(ingredients, cuisine)
    print(recipe)
