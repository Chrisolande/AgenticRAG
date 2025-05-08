import os
from pprint import pprint

from graph import create_graph, init_globals

def setup_environment():
    
    # Create vector directory if it doesn't exist
    os.makedirs("vector", exist_ok=True)

def process_query(query):
    """
    Process a query through the RAG workflow

    """
    # Create input for the workflow
    inputs = {"question": query}
    
    # Create and run the graph
    app = create_graph()
    
    # Stream outputs for debugging
    final_output = None
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
        final_output = value
    
    # Return the final generation
    return final_output["generation"]

def main():
    """
    Main function
    """
    # Set up environment
    setup_environment()
    
    # Initialize global components
    init_globals()
    
    # Continuously prompt the user until they type 'exit' or 'quit'
    while True:
        query = input("\nEnter your question (type 'exit' or 'quit' to end): ")
        
        # Check if user wants to exit
        if query.lower() in ['exit', 'quit']:
            print("Exiting program. Goodbye!")
            break
        
        # Check if input is empty or just whitespace
        if not query.strip():
            print("Please enter a valid question.")
            continue
        
        # Process the query and display response
        response = process_query(query)
        
        # Print the response
        print("\nFinal response:")
        print(response)

if __name__ == "__main__":
    main()
