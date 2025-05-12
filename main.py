import argparse
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from agent.simple_agent import SimpleAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Claude Plays Pokemon - Starter Version")
    parser.add_argument(
        "--rom", 
        type=str, 
        default="pokemon.gb",
        help="Path to the Pokemon ROM file"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=10, 
        help="Number of agent steps to run"
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Run with display (not headless)"
    )
    parser.add_argument(
        "--sound", 
        action="store_true", 
        help="Enable sound (only applicable with display)"
    )
    parser.add_argument(
        "--max-history", 
        type=int, 
        default=30, 
        help="Maximum number of messages in history before summarization"
    )
    parser.add_argument(
        "--load-state", 
        type=str, 
        default=None, 
        help="Path to a saved state to load"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode (wait for user input after each step)"
    )
    
    args = parser.parse_args()
    
    # Get absolute path to ROM
    if not os.path.isabs(args.rom):
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rom)
    else:
        rom_path = args.rom
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return
    
    # Create and run agent
    agent = SimpleAgent(
        rom_path=rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
        max_history=args.max_history,
        load_state=args.load_state,
        interactive=args.interactive,
    )
    
    try:
        logger.info(f"Starting agent for {args.steps} steps")
        steps_completed = agent.run(num_steps=args.steps)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
    finally:
        agent.stop()

if __name__ == "__main__":
    main()