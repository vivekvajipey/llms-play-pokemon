# Claude Plays Pokemon - Starter Version

A minimal implementation of Claude playing Pokemon Red using the PyBoy emulator. This starter version includes:

- Simple agent that uses Claude to play Pokemon Red
- Memory reading functionality to extract game state information
- Basic emulator control through Claude's function calling

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in the `.env` file:
   ```
   # Copy the template file
   cp .env.template .env
   
   # Edit the .env file and add your API key
   # OPENAI_API_KEY=your_api_key_here
   ```

4. Place your Pokemon Red ROM file in the root directory (you need to provide your own ROM)

## Usage

Run the main script:

```
python main.py
```

Optional arguments:
- `--rom`: Path to the Pokemon ROM file (default: `pokemon.gb` in the root directory)
- `--steps`: Number of agent steps to run (default: 10)
- `--display`: Run with display (not headless)
- `--sound`: Enable sound (only applicable with display)

Example:
```
python main.py --rom pokemon.gb --steps 20 --display --sound
```

## Implementation Details

### Components

- `agent/simple_agent.py`: Main agent class that uses Claude to play Pokemon
- `agent/emulator.py`: Wrapper around PyBoy with helper functions
- `agent/memory_reader.py`: Extracts game state information from emulator memory

### How It Works

1. The agent captures a screenshot from the emulator
2. It reads the game state information from memory
3. It sends the screenshot and game state to Claude
4. Claude responds with explanations and emulator commands
5. The agent executes the commands and repeats the process