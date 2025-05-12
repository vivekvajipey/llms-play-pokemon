#!/bin/bash

# Check if the ROM file exists
if [ ! -f "pokemon.gb" ]; then
  echo "Error: pokemon.gb ROM file not found in the current directory."
  echo "Please place your Pokemon ROM file in this directory and name it 'pokemon.gb'."
  exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
  echo "Warning: .env file not found. Creating a template .env file..."
  echo "# Add your API key below" > .env
  echo "OPENAI_API_KEY=your_api_key_here" >> .env
  echo ""
  echo "Please edit the .env file and add your OpenAI API key before running again."
  exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Run the game with display and sound, and forward any additional arguments
python main.py --display --sound --steps 100 "$@"

# Deactivate the virtual environment when done
deactivate
