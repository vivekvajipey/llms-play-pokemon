"""
Simple test script to verify that LiteLLM is working correctly with the Fireworks API key.
"""
import os
from dotenv import load_dotenv
from litellm import completion

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("FIREWORKS_API_KEY")
if not api_key:
    raise ValueError("FIREWORKS_API_KEY not found in environment variables")

print(f"API key found: {api_key[:5]}...{api_key[-4:]}")

# Define the model to use
model = "fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic"
print(f"Using model: {model}")

# Set the API key in the environment
os.environ["FIREWORKS_AI_API_KEY"] = api_key

# Test a simple completion
try:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and tell me what model you are."}
        ],
        max_tokens=100
    )
    
    # Print the response
    print("\nResponse from LiteLLM:")
    print(f"Content: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
    print("\nTest successful! LiteLLM is working correctly with your Fireworks API key.")
    
except Exception as e:
    print(f"\nError: {e}")
    print("\nTest failed. Please check your API key and model configuration.")
