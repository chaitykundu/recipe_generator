from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_full_recipe(dish_name: str, cuisine: str = "Any"):
    messages = [
        {"role": "system", "content": "You are a helpful recipe expert who writes complete recipes clearly."},
        {"role": "user", "content": f"""
I want to cook the dish: {dish_name}
Create a {cuisine} style recipe
Provide a complete Bangladeshi-style recipe for this dish. Include:
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
            max_new_tokens=350,
            temperature=0.8,
            top_k=50,
            do_sample=True
        )

    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return reply.strip()

# Example usage
if __name__ == "__main__":
    dish = input("🍽️ Enter the dish name: ")
    cuisine = input("Enter cuisine (default: Any): ") or "Any"
    recipe = generate_full_recipe(dish, cuisine)
    print("\n🧑‍🍳 Generated Recipe:\n")
    print(recipe)
