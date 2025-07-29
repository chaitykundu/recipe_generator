import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the model and tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#MODEL_NAME = "nvidia/OpenReasoning-Nemotron-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_full_recipe(dish_name: str, cuisine: str = "Any"):

    if not dish_name.replace(" ", "").isalpha():
        return "Invalid dish name. Please enter a real dish name."

    messages = [
        {"role": "system", "content": "You are a helpful recipe expert who writes complete recipes clearly."},
        {"role": "user", "content": f"""
I want to cook the dish: {dish_name}
Provide a complete {cuisine} style recipe for this dish. Include:
- A list of required ingredients
- Estimated cooking time
- Step-by-step cooking instructions (written clearly and in order)

Respond in structured format and do not skip any part. Keep instructions practical and easy to follow.
"""}
    ]

    # Tokenize using chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            top_k=50,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return reply.strip()

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
    

# Example usage
if __name__ == "__main__":
    print("Welcome to my AI Recipe Generator!")

    mode = input("Choose input mode (1 = Text, 2 = Voice): ")

    if mode.strip() == "2":
        dish_name = listen_microphone("Say the dish_name:")
        cuisine = listen_microphone("Say your preferred cuisine or say 'Any':")
        if not cuisine.strip():
            cuisine = "Any"
    else:
        dish_name = input("Enter dish_name: ")
        cuisine = input("Enter preferred cuisine (e.g., Bangladeshi, Italian) or leave empty: ")
        if not cuisine.strip():
            cuisine = "Any"

    #dish = input("Enter the dish name: ")
    #cuisine = input("Enter cuisine (default: Any): ") or "Any"
    recipe = generate_full_recipe(dish_name, cuisine)
    print("\n🧑‍🍳 Generated Recipe:\n")
    print(recipe)




