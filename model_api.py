from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned model and tokenizer
base_model = "aboutaleb/llama-3-8b-chat-tourismneweeest"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load the model in CPU mode without quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    torch_dtype=torch.float32,  # Using float32 for CPU
    device_map="cpu",           # Force the model to load on CPU
)

# Set the model to evaluation mode
model.eval()

# Define the input structure for the API request
class TextGenerationRequest(BaseModel):
    content: str

# API endpoint for generating text
@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    # Example conversation prompt
    messages = [{"role": "user", "content": request.content}]
    
    # Create the prompt using the chat template from tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Set up the text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu"  # Specify CPU
    )
    
    # Generate the response
    outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    # Return the generated text
    return {"response": outputs[0]["generated_text"]}
