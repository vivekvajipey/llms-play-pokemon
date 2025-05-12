import base64
import copy
import io
import json
import logging
import os
import copy

# Set environment variable to disable LiteLLM logs
os.environ['LITELLM_LOG'] = 'ERROR'

from config import (
    MAX_TOKENS, MODEL_NAME, TEMPERATURE, USE_NAVIGATOR,
    API_TYPE, REASONING_EFFORT, REASONING_SUMMARY
)

from agent.emulator import Emulator
import litellm
from litellm import completion
import openai
from dotenv import load_dotenv

# Completely disable LiteLLM debugging
litellm._logging._disable_debugging()

# Load environment variables from .env file
load_dotenv()

# Set the API keys based on the selected API type
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # Also set for LiteLLM in case we're using it
    if API_TYPE == "litellm":
        os.environ["LITELLM_OPENAI_API_KEY"] = openai_api_key

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress all LiteLLM logs using multiple approaches
# 1. Set the logger level to higher than CRITICAL
for logger_name in ["LiteLLM", "LiteLLM Proxy", "LiteLLM Router"]:
    litellm_logger = logging.getLogger(logger_name)
    litellm_logger.setLevel(logging.CRITICAL + 1)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the emulator tool to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


# LiteLLM tool format
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "press_buttons",
            "description": "Press a sequence of buttons on the Game Boy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "buttons": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["a", "b", "start", "select", "up", "down", "left", "right"]
                        },
                        "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Whether to wait for a brief period after pressing each button. Defaults to true."
                    }
                },
                "required": ["buttons"]
            }
        }
    }
]

# Function to convert LiteLLM tool format to OpenAI Responses API format
def convert_litellm_to_openai(tools_lite):
    openai_tools = []
    for t in tools_lite:
        # only functions for now
        if t.get("type") == "function":
            fn = t["function"]
            openai_tools.append({
                "type": "function",
                "name": fn["name"],
                "description": fn["description"],
                "parameters": fn["parameters"],
            })
    return openai_tools

# Function to convert chat history to Responses API format
def history_for_responses(msgs):
    """Convert internal chat-style history to Responses API format."""
    converted = []
    for m in msgs:
        # Handle assistant messages that contain tool_calls
        if m.get("role") == "assistant" and m.get("tool_calls"):
            # Keep the assistant's content as a regular message
            if m.get("content"):
                converted.append({
                    "role": "assistant",
                    "content": m["content"]
                })
            
            # Add each tool call as a function_call item
            for tc in m["tool_calls"]:
                # Ensure arguments is a JSON string
                arg_str = tc["function"]["arguments"]
                if not isinstance(arg_str, str):
                    arg_str = json.dumps(arg_str)
                    
                converted.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": arg_str,  # Must be a JSON string
                })
        # Handle tool role messages (function call outputs)
        elif m.get("role") == "tool":
            # Ensure output is a string
            out_str = m["content"]
            if not isinstance(out_str, str):
                out_str = json.dumps(out_str)
                
            converted.append({
                "type": "function_call_output",
                "call_id": m["tool_call_id"],
                "output": out_str,  # Must be a string
            })
        # Regular chat messages stay unchanged
        else:
            converted.append(m)
    return converted

# Generate OpenAI Responses API tool format from LiteLLM format
OPENAI_TOOLS = convert_litellm_to_openai(AVAILABLE_TOOLS)

if USE_NAVIGATOR:
    # Add navigator tool to LiteLLM format
    AVAILABLE_TOOLS.append({
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Navigate to a specific position on the map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "The row coordinate to navigate to (0-8)."
                    },
                    "col": {
                        "type": "integer",
                        "description": "The column coordinate to navigate to (0-9)."
                    }
                },
                "required": ["row", "col"]
            }
        }
    })
    
    # Regenerate OpenAI tools after adding the navigator tool
    OPENAI_TOOLS = convert_litellm_to_openai(AVAILABLE_TOOLS)


class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, load_state=None, interactive=False):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            load_state: Path to a saved state to load
            interactive: Whether to run in interactive mode (wait for user input after each step)
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()  # Initialize the emulator
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        self.interactive = interactive
        
        if load_state:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)

    def process_tool_call(self, tool_call):
        """Process a single tool call."""
        # Handle the new function format from LiteLLM
        if hasattr(tool_call, 'function'):
            # New format from LiteLLM
            tool_name = tool_call.function.name
            # Properly parse JSON arguments instead of using eval
            tool_input = json.loads(tool_call.function.arguments)
        else:
            # Original format
            tool_name = tool_call.name
            tool_input = tool_call.input
            
        logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")
            
            result = self.emulator.press_buttons(buttons, wait)
            
            # Get a fresh screenshot after executing the buttons
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after your button presses:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            logger.info(f"[Navigation] Navigating to: ({row}, {col})")
            
            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get a fresh screenshot after executing the navigation
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            
            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")
            
            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Navigation result: {result}"},
                    {"type": "text", "text": "\nHere is a screenshot of the screen after navigation:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                ],
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                messages = copy.deepcopy(self.message_history)

                if len(messages) >= 3:
                    if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                    
                    if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                        messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

                # Get response using either LiteLLM or direct OpenAI API
                if API_TYPE == "litellm":
                    # Use LiteLLM for the completion
                    response = completion(
                        model=MODEL_NAME,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        tools=AVAILABLE_TOOLS
                    )
                    
                    logger.info(f"Response usage: {response.usage}")
                    
                    # Get the message content and tool calls
                    message = response.choices[0].message
                    content = message.content
                    tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else []
                    
                    # No reasoning summary available with LiteLLM
                    reasoning_summary = None
                    
                else:  # API_TYPE == "openai"
                    # Use OpenAI's Responses API directly to get reasoning summaries
                    client = openai.OpenAI()
                    
                    # Prepare the input for the Responses API
                    system_message = {"role": "system", "content": SYSTEM_PROMPT}
                    
                    # Convert the message history to Responses API format
                    converted_messages = history_for_responses(messages)
                    all_messages = [system_message] + converted_messages
                    
                    # Format the input as a simple array of messages for the Responses API
                    response = client.responses.create(
                        model=MODEL_NAME,
                        input=all_messages,  # Converted messages in Responses API format
                        tools=OPENAI_TOOLS,  # Use OpenAI-specific tool format
                        reasoning={
                            "effort": REASONING_EFFORT,
                            "summary": REASONING_SUMMARY
                        }
                    )
                    
                    # Get the message content and tool calls
                    content = ""
                    tool_calls = []
                    reasoning_summary = None
                    
                    # Process each output item
                    for item in response.output:
                        # Extract message content
                        if item.type == "message":
                            # Get the message content
                            for content_item in item.content:
                                if content_item.type == "output_text":
                                    content += content_item.text
                        
                        # Extract function call if available
                        elif item.type == "function_call":
                            # Create a tool call object compatible with the rest of the code
                            tool_call = type('ToolCall', (), {})()
                            tool_call.id = item.call_id
                            tool_call.function = type('Function', (), {})()
                            tool_call.function.name = item.name
                            tool_call.function.arguments = item.arguments
                            tool_calls.append(tool_call)
                        
                        # Extract reasoning summary if available
                        elif item.type == "reasoning" and hasattr(item, "summary") and item.summary:
                            reasoning_summary = item.summary[0].text
                            logger.info(f"[Reasoning Summary] {reasoning_summary}")
                
                # Display the model's reasoning
                logger.info(f"[Text] {content}")
                
                if tool_calls:
                    for tool_call in tool_calls:
                        logger.info(f"[Tool] Using tool: {tool_call.function.name}")

                # Process tool calls
                if tool_calls:
                    # Log the tool calls for debugging
                    for tc in tool_calls:
                        logger.info(f"Tool call: id={tc.id}, name={tc.function.name}")
                        logger.info(f"Arguments: {tc.function.arguments}")
                    
                    # Add LLM's response to history
                if content or tool_calls:
                    # Create the assistant message with content and tool calls
                    assistant_message = {
                        "role": "assistant",
                        "content": content
                    }
                    
                    # Add tool calls if any
                    if tool_calls:
                        assistant_message["tool_calls"] = [{
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in tool_calls]
                    
                    # Add the assistant message to history
                    self.message_history.append(assistant_message)
                
                # Process tool calls and create tool results
                tool_results = []
                for tool_call in tool_calls:
                    # Process the tool call directly without creating an adapted version
                    tool_result = self.process_tool_call(tool_call)
                    tool_results.append(tool_result)
                
                # Add tool results to message history
                for i, tool_call in enumerate(tool_calls):
                    tool_result_text = ""
                    # Extract text content from the tool result
                    for content_item in tool_results[i].get('content', []):
                        if content_item.get('type') == 'text':
                            tool_result_text += content_item.get('text', '') + "\n"
                    
                    # Add as a tool message with the specific tool_call_id
                    self.message_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_text.strip()
                    })
                    
                    # Also add a user message to continue the conversation
                    self.message_history.append({
                        "role": "user",
                        "content": "What's your next move based on what you see?"
                    })

                    # Check if we need to summarize the history
                    if len(self.message_history) >= self.max_history:
                        self.summarize_history()

                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")
                
                # Add breakpoint - wait for user input to continue if in interactive mode
                if self.interactive:
                    user_input = input("Press Enter to continue to the next step (or 'q' to quit): ")
                    if user_input.lower() == 'q':
                        logger.info("User requested to quit")
                        self.running = False

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                raise e

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info(f"[Agent] Generating conversation summary...")
        
        # Get a new screenshot for the summary
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        
        # Create messages for the summarization request - pass the entire conversation history
        messages = copy.deepcopy(self.message_history) 


        if len(messages) >= 3:
            if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            
            if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SUMMARY_PROMPT,
                    }
                ],
            }
        ]
        
        # Get summary using either LiteLLM or direct OpenAI API
        if API_TYPE == "litellm":
            # Use LiteLLM for the summary
            response = completion(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            # Extract the summary text
            summary_text = response.choices[0].message.content
            
            # No reasoning summary available with LiteLLM
            reasoning_summary = None
            
        else:  # API_TYPE == "openai"
            # Use OpenAI's Responses API directly to get reasoning summaries
            client = openai.OpenAI()
            
            # Prepare the input for the Responses API
            system_message = {"role": "system", "content": SYSTEM_PROMPT}
            
            # Convert the message history to Responses API format
            converted_messages = history_for_responses(messages)
            all_messages = [system_message] + converted_messages
            
            # Format the input as a simple array of messages for the Responses API
            response = client.responses.create(
                model=MODEL_NAME,
                input=all_messages,  # Converted messages in Responses API format
                reasoning={
                    "effort": REASONING_EFFORT,
                    "summary": REASONING_SUMMARY
                }
            )
            
            # Extract the summary text and reasoning summary
            summary_text = ""
            reasoning_summary = None
            
            # Process each output item
            for item in response.output:
                # Extract message content
                if item.type == "message":
                    # Get the message content
                    for content_item in item.content:
                        if content_item.type == "output_text":
                            summary_text += content_item.text
                
                # Extract reasoning summary if available
                elif item.type == "reasoning" and hasattr(item, "summary") and item.summary:
                    reasoning_summary = item.summary[0].text
                    logger.info(f"[Summarization Reasoning] {reasoning_summary}")
                    
            if not summary_text:
                # Fallback in case we couldn't extract the summary text
                logger.warning("Could not extract summary text from OpenAI response, using default")
                summary_text = "Could not generate summary. Continuing with the game."
        
        logger.info(f"[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")
        
        # Create a combined text message with the summary
        combined_text = f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}\n\n"
        combined_text += "You were just asked to summarize your playthrough so far, which is the summary you see above.\n"
        combined_text += "Here is the current game state. You may now continue playing by selecting your next action."
        
        # Replace message history with just the summary and a new screenshot
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": combined_text
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    }
                ]
            }
        ]
        
        logger.info(f"[Agent] Message history condensed into summary.")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()