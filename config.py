# Configuration for the application

# API Configuration
# Set to 'litellm' to use LiteLLM or 'openai' to use OpenAI's API directly with reasoning summaries
API_TYPE = "openai"

# Model Configuration
MODEL_NAME = "o4-mini"  # Used by both LiteLLM and OpenAI
TEMPERATURE = 1.0
MAX_TOKENS = 4000

# OpenAI Reasoning Configuration (only used when API_TYPE = "openai")
REASONING_EFFORT = "low"  # Options: "low", "medium", "high"
REASONING_SUMMARY = "detailed"   # Options: "auto", "detailed", "concise", or None

# Set to True to enable the navigation tool
USE_NAVIGATOR = False