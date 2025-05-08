import os
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
def process_query(query):
    """
    Process a query through the RAG workflow and return the generated response.
    Args:
        query (str): The user's input question.
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
            pprint(f"Finished running: {key}:")
        final_output = value
    
    # Return the final generated response
    return final_output["generation"]

# Main function to run the application
def main():
    """
    Main function to initialize the application and handle user interaction.
    """
    # Set up the environment
    setup_environment()
    
    # Initialize global components required for the workflow
    init_globals()
    
    # Continuously prompt the user until they type 'exit' or 'quit'
    while True:
        query = input("\nEnter your question (type 'exit' or 'quit' to end): ")
        
        # Check if the user wants to exit
        if query.lower() in ['exit', 'quit']:
            print("Exiting program. Goodbye!")
            break
        
        # Check if the input is empty or just whitespace
        if not query.strip():
            print("Please enter a valid question.")
            continue
        
        # Process the query and display the response
        response = process_query(query)
        
        # Print the final response
        print("\nFinal response:")
        print(response)

# Entry point for the script
if __name__ == "__main__":
    main()
