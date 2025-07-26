from fastapi import FastAPI
from pydantic import BaseModel
from model import generate_recipe

app = FastAPI()

# Root route to verify app is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Personalized Recipe AI!"}

# Input schema
class RecipeRequest(BaseModel):
    ingredients: str
    cuisine: str = "Any"

# POST /generate to get recipe
@app.post("/generate")
def recipe(req: RecipeRequest):
    result = generate_recipe(req.ingredients, req.cuisine)
    return {"recipe": result}
