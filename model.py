from transformers import pipeline

generator = pipeline("text-generation", model="sentence-transformers/all-MiniLM-L6-v2")

def generate_recipe(ingredients, cuisine):
    prompt = f"""
    I have these ingredients: {ingredients}
    Create a {cuisine} style recipe. Include:
    - Dish name
    - Estimated cooking time
    - Step-by-step instructions
    """
    result = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]
