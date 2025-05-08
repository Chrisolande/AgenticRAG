import os
import argparse
from pprint import pprint

from graph import create_graph, init_globals

# Function to set up the environment
def setup_environment():
    """
    Ensure necessary directories exist for the application.
    """
    # Create vector directory if it doesn't exist
    os.makedirs("vector", exist_ok=True)

# Function to process a user query through the RAG workflow
def process_query(query, verbose=False):
    """
    Process a query through the RAG workflow and return the generated response.
    Args:
        query (str): The user's input question.
        verbose (bool): Whether to print intermediate outputs.
    Returns:
        str: The final generated response.
    """
    # Create input for the workflow
    inputs = {"question": query}
    
    # Create and run the graph workflow
    app = create_graph()
    
    # Stream outputs for debugging and capture the final output
    final_output = None
    for output in app.stream(inputs):
        for key, value in output.items():
            if verbose:
                pprint(f"Finished running: {key}:")
        final_output = value
    
    # Return the final generated response
    return final_output["generation"]

# Interactive mode function
def interactive_mode(verbose=False):
    """
    Run the application in interactive mode, continuously prompting for input.
    Args:
        verbose (bool): Whether to print intermediate outputs.
    """
    print("Welcome to AgenticRAG Interactive Mode!")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Continuously prompt the user until they type 'exit' or 'quit'
    while True:
        query = input("\nEnter your question: ")
        
        # Check if the user wants to exit
        if query.lower() in ['exit', 'quit']:
            print("Exiting program. Goodbye!")
            break
        
        # Check if the input is empty or just whitespace
        if not query.strip():
            print("Please enter a valid question.")
            continue
        
        # Process the query and display the response
        response = process_query(query, verbose)
        
        # Print the final response
        print("\nFinal response:")
        print(response)

# Main function to run the application
def main():
    """
    Main function to initialize the application and handle user interaction.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="AgenticRAG - Retrieval-Augmented Generation System for Text Analysis and Response",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        "--query", "-q", 
        type=str,
        help="Run a single query and exit"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Display intermediate processing steps"
    )
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Run in interactive mode (default if no query provided)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up the environment
    setup_environment()
    
    # Initialize global components required for the workflow
    init_globals()
    
    # Determine mode of operation
    if args.query:
        # Single query mode
        response = process_query(args.query, args.verbose)
        print(response)
    else:
        # Interactive mode (default)
        interactive_mode(args.verbose)

# Entry point for the script
if __name__ == "__main__":
    main()
