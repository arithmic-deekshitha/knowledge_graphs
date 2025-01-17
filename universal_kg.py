
import os
import getpass
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyvis.network import Network
from typing import List, Set
from dataclasses import dataclass
import json

# Initialize Ollama LLM
llm = Ollama(model="llama3.3:70b-instruct-q4_K_M", base_url="http://localhost:11434")

# Initialize Graph Transformer
llm_transformer = LLMGraphTransformer(llm=llm)

def extract_info_direct(data, prefix=""):
    """
    Recursively extract and format JSON fields into text.
    """
    text = ""
    if isinstance(data, dict):
        for key, value in data.items():
            # Recursively process dictionary values
            text += f"{prefix}{key.capitalize()}: {value if not isinstance(value, (dict, list)) else ''}\n"
            text += extract_info_direct(value, prefix=prefix + "  ")
    elif isinstance(data, list):
        for i, item in enumerate(data, start=1):
            text += f"{prefix}- Item {i}:\n"
            text += extract_info_direct(item, prefix=prefix + "  ")
    return text


def extract_info(data, user, parent_key="", prefix=""):
    """
    Recursively extract and format JSON fields into text,
    while maintaining relationships using the parent_key and appending the user string.
    """
    text = ""
    
    if isinstance(data, dict):
        for key, value in data.items():
            # Only append 'user' at the root level for the first time
            current_key = f"{user}'s {key}" if not parent_key else f"{parent_key}'s {key}"
            if not isinstance(value, (dict, list)):
                text += f"{prefix}{current_key}: {value}\n"
            else:
                text += f"{prefix}{current_key}:\n"
            text += extract_info(value, user=user, parent_key=current_key, prefix=prefix + "  ")
    
    elif isinstance(data, list):
        for i, item in enumerate(data, start=1):
            # Only append 'user' once at the root, avoid repeating it in the list items
            current_key = f"{parent_key}'s item {i}" if parent_key else f"{user}'s item {i}"
            text += f"{prefix}{current_key}:\n"
            text += extract_info(item, user=user, parent_key=current_key, prefix=prefix + "  ")

    return text


def process_json_to_graph(user_info: str, json_data: str, chunk_size: int = 2000, chunk_overlap: int = 200):
    """
    Process a long text into a knowledge graph by splitting it into chunks
    and combining the results.
    """
    # Create text splitter

    text = extract_info(json_data, user_info)
    print(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Split text into chunks
    documents = text_splitter.create_documents([text])
    print(f"Split text into {len(documents)} chunks")
    
    # Process each chunk
    all_nodes = []
    all_relationships = []
    
    for i, doc in enumerate(documents):
        print(f"Processing chunk {i+1}/{len(documents)}")
        try:
            # combined_text = user_info + "\n" + doc.page_content
            # combined_document = Document(page_content=combined_text)
            graph_doc = llm_transformer.convert_to_graph_documents([doc])[0]
            all_nodes.extend(graph_doc.nodes)
            all_relationships.extend(graph_doc.relationships)
            return deduplicate(all_nodes, all_relationships)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
def deduplicate(all_nodes, all_relationships):
    # Deduplicate nodes by ID
    unique_nodes = list({node.id: node for node in all_nodes}.values())
    
    # Deduplicate relationships (considering source, target, and type)
    seen_relationships = set()
    unique_relationships = []
    
    for rel in all_relationships:
        rel_tuple = (rel.source.id, rel.target.id, rel.type)
        if rel_tuple not in seen_relationships:
            seen_relationships.add(rel_tuple)
            unique_relationships.append(rel)
    
    print(f"Final graph contains {len(unique_nodes)} nodes and {len(unique_relationships)} relationships")
    
    return unique_nodes, unique_relationships

def create_interactive_graph(nodes, relationships, filename: str = "combined_knowledge_graph.html"):
    """
    Create and save an interactive visualization of the knowledge graph.
    """
    # Initialize the network
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#ffffff", font_color="#000000")
    
    # Add nodes
    for node in nodes:
        net.add_node(node.id, label=node.id, title=node.id)
    
    # Add edges
    for rel in relationships:
        net.add_edge(rel.source.id, rel.target.id, 
                    label=rel.type, 
                    title=f"{rel.source.id} -> {rel.type} -> {rel.target.id}")
    
    # Configure physics layout
    net.set_options("""
    const options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "springLength": 250,
                "springConstant": 0.5
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
        }
    }
    """)
    
    # Save the graph
    net.show(filename)
    print(f"Graph saved as {filename}")


def generate_recommendation(kg: dict[str, List[dict]], user_context: [str] = "Give me recommendations based on the kg") -> str:
    """
    Generate a recommendation using a KG and an optional user context.
    """
    nodes = kg.get("nodes", [])
    relationships = kg.get("relationships", [])
    
    # Generate a textual representation of the KG
    kg_description = "The knowledge graph contains the following:\n"
    kg_description += "\n".join([f"Node: {node.id}" for node in nodes])
    kg_description += "\nRelationships:\n"
    kg_description += "\n".join([f"{rel.source.id} -> {rel.type} -> {rel.target.id}" for rel in relationships])
    
    # Include user context if provided
    if user_context:
        prompt = f"{user_context}. {kg_description}"
    else:
        prompt = kg_description

    prompt = f"""
    You are a personalized recommendation agent designed to provide tailored suggestions and answers based on a user-specific knowledge graph. The knowledge graph is constructed from the user's data, and your job is to analyze it meticulously to deliver actionable, insightful, and relevant recommendations.
When responding, ensure that your suggestions are precise, thoughtful, and directly address the user's query. Always align your recommendations with the user's preferences, interests, and goals as inferred from the knowledge graph.
Here is the query you need to address:
{prompt}
Based on the knowledge graph, generate a response that is clear, specific, and valuable to the user.
    """

    # print("the promt:\n", prompt)
    # Generate recommendation using the Llama model
    recommendation = llm.invoke(prompt)
    return recommendation

def chatbot(kg: dict[str, List[dict]]):
    """
    A chatbot that interacts with the user and generates recommendations
    based on the knowledge graph (kg) and user context.
    It will continue to ask for prompts until 'quit' is typed.
    """
    while True:
        user_context = input("Ask me anything (type 'quit' to exit)\n You: ")
        
        if user_context.lower() == "quit":
            print("Exiting chatbot...")
            break
        
        # Process the user's input prompt and generate recommendation
        print("\nGenerating recommendation based on your input...\n")
        recommendation = generate_recommendation(kg, user_context)
        
        # Output the recommendation
        print("\nResponse: \n", recommendation)


# Example usage
if __name__ == "__main__":
    json_filenames = ["sample_twt_profile.json", "sample_twt_profile1.json"]
    combined_nodes =[]
    combined_relationships = []
    for json_filename in json_filenames:
        with open(json_filename, "r") as file:
            json_data = json.load(file)
        user_info = f"""{json_data.get("username")}"""
        nodes, relationships = process_json_to_graph(user_info=user_info, json_data=json_data)
        combined_nodes.extend(nodes)
        combined_relationships.extend(relationships)
        
    nodes, relationships = deduplicate(combined_nodes, combined_relationships)
    # Create and save the visualization
    create_interactive_graph(nodes, relationships)
    # print("\n response: \n",generate_recommendation({"nodes": nodes, "relationships": relationships}, "give me very specific recommendation based on the information provided about my twitter. like what accounts i could follow, what i can tweet more about, any specific news i need to keep my eye on. Be Very Specific in the recommendations. "))    
    chatbot({"nodes": nodes, "relationships": relationships})