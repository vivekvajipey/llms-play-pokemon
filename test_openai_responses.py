#!/usr/bin/env python3
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

print(f"API key found: {api_key[:5]}...{api_key[-4:]}")

# Define the model to use
model = "o4-mini"
print(f"Using model: {model}")

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = api_key

# Test a simple completion with the Responses API
try:
    client = openai.OpenAI()
    
    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello and tell me what model you are."}
    ]
    
    # Use the Responses API
    response = client.responses.create(
        model=model,
        input=messages,
        reasoning={
            "effort": "low",
            "summary": "detailed"
        }
    )
    
    # Print the response
    print("\nResponse:")
    print(f"Raw response: {response}")
    
    # Print the response structure to understand its format
    print("\nResponse structure:")
    for attr in dir(response):
        if not attr.startswith('_'):
            print(f"{attr}: {getattr(response, attr)}")
    
    # Try to extract the content
    try:
        if hasattr(response, 'output'):
            print("\nOutput:")
            print(response.output)
            
            if hasattr(response.output, 'items'):
                for i, item in enumerate(response.output.items):
                    print(f"\nItem {i}:")
                    print(f"Type: {item.type}")
                    
                    if item.type == 'message':
                        print("Message content:")
                        if hasattr(item, 'message') and hasattr(item.message, 'content'):
                            for content_item in item.message.content:
                                print(f"Content: {content_item.text}")
                    
                    if item.type == 'reasoning':
                        print("Reasoning:")
                        if hasattr(item, 'summary'):
                            for summary_item in item.summary:
                                print(f"Summary: {summary_item.text}")
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    print("\nSuccess!")
    
except Exception as e:
    print(f"Error: {e}")
