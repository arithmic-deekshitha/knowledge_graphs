# import os
# import getpass

# from langchain_experimental.graph_transformers import LLMGraphTransformer
# # from langchain.llms import Ollama
# from langchain_community.llms import Ollama
# from langchain_core.documents import Document
# #from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# # Initialize Ollama LLM
# # llm = Ollama(model="mistral:7b-instruct-q4_K_M", base_url="http://localhost:11434")
# llm = Ollama(model="llama3.3:70b-instruct-q4_K_M", base_url="http://localhost:11434")

# #mistral:7b-instruct-q4_K_M
# #llama3.3:70b-instruct-q4_K_M

# # Initialize Graph Transformer with Ollama LLM
# llm_transformer = LLMGraphTransformer(llm=llm)

# # text = """
# # Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
# # She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
# # Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
# # She was, in 1906, the first woman to become a professor at the University of Paris.
# # """

# text = """
# Review
# Overview of the public and private health sectors 

# The government-funded health sector, which is the provider of healthcare to vulnerable populations, has been chronically underfunded with 1.28% of the GDP. This translates to a healthcare expenditure of $2.7 per citizen per year. As a consequence, India has 0.7 public hospital beds per 100,000 people [2] now and 0.576 physicians per 1,000 population in 2000 [4], compared to the World Health Organization's recommended doctor-to-population ratio of 1:1,000 [5]. Since the inception of the National Health Mission (NHM) in 2005, the government has aimed to increase the quantum of services provided, but a lack of focus on quality has failed to make a dent in healthcare indicators [6]. At best, 37% of the population had any health insurance coverage in 2018 [7].

# This has contributed to the for-profit private health sector becoming the dominant provider of healthcare for it is perceived to provide quality care [8]. It consumes 5.1% of the GDP, which is financed by Out-Of-Pocket (OOP) expenditure. This sector spans a wide range, from world-class health facilities, such as Narayana Health, an internationally accredited, high quality, tertiary healthcare service provider, to individual informal provider clinics, which are establishments providing medical care, often manned by a solo provider who does not have a formal medical qualification or registration. World-class health facilities exist in urban areas and have enabled India to become a leading destination for medical tourism [9]. The informal providers are concentrated in urban slums and rural areas where they are the first choice of care for they have built a long-standing trusted constant presence in their communities and have adapted to their social, economic, and cultural norms. 
# Public health sector (NHM) strategies

# The NHM has sought to address health challenges through five approaches - communitization, flexible financing, improved management through capacity building, monitoring progress against standards, and innovations in human resource management [10]. Multiple new cadres have been created to provide primary healthcare and accelerate the pace toward universal health coverage. 

# A key cadre has been the female community health workers, Accredited Social Health Activists (ASHA), one for every 1,000 population. They number a million [11] to date and serve to increase the reach of the Auxiliary Nurse Midwives (ANM), who were meant to serve a population of 5,000 but in reality, serve up to 20,000 people. This exercise in task shifting from ANMs to ASHAs cadre raises concerns. ANMs are high school graduates who receive 18 months of training while ASHAs attend school up to the eighth grade, sometimes even less, and receive 23 days’ initial training with additional on-the-job need-based short training [12]. ANMs are government employees and receive a salary while the ASHAs are treated as private contractors who receive a payment proportionate to the amount of work performed. A recent synthesis of the evaluation of ASHAs from a health systems perspective shows broader system constraints and few overall positive findings [12].

# Other new cadres are being hired by the government on a contractual basis. the Rural Medical Assistants (RMAs) and paramedical personnel. The RMA [13] is a three-year diploma [13,14] course similar to the physician assistant program in the US. The RMAs work at Primary Health Centers and initial evaluation [15] is positive. Several paramedical councils now exist which have a number of courses varying in duration from six to 24 months to train personnel for the provision of supportive care in health facilities and homes. The scale of such initiatives remains nascent and systemic integration pathways remain undefined. 
# The recent increase in the federal health budget offers an unprecedented opportunity to do this. This article utilizes the ready materials, extract and analyze data, distill findings (READ) approach to adding to the authors' experiential learning to analyze the health system in India. The growing divide between the public and the burgeoning private health sector systems, with the latter's booming medical tourism industry and medical schools, are analyzed along with the newly minted National Medical Council, to recommend policies that would help India achieve its SDGs.
# """

# documents = [Document(page_content=text)]
# graph_documents = llm_transformer.convert_to_graph_documents(documents)
# print(f"Nodes:{graph_documents[0].nodes}")
# print(f"Relationships:{graph_documents[0].relationships}")

# from pyvis.network import Network

# # Initialize the interactive netfwork
# net = Network(notebook=True, height="750px", width="100%")

# # Add nodes
# for node in graph_documents[0].nodes:
#     net.add_node(node.id, label=node.id)

# # Add edges
# for rel in graph_documents[0].relationships:
#     net.add_edge(rel.source.id, rel.target.id, label=rel.type)

# # Display the graph
# net.show("knowledge_graph_1.html")


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
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
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

    prompt = f""" You are a personalized recommendation agent designed to provide tailored suggestions and answers based on a user-specific knowledge graph. The knowledge graph is constructed from the user's data, and your job is to analyze it meticulously to deliver actionable, insightful, and relevant recommendations.
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
    json_filename = "sample_twt_profile.json"
    with open(json_filename, "r") as file:
        json_data = json.load(file)
    user_info = f"""{json_data.get("username")}"""

    nodes, relationships = process_json_to_graph(user_info=user_info, json_data=json_data)
    
    # Create and save the visualization
    create_interactive_graph(nodes, relationships)
    # print("\n response: \n",generate_recommendation({"nodes": nodes, "relationships": relationships}, "give me very specific recommendation based on the information provided about my twitter. like what accounts i could follow, what i can tweet more about, any specific news i need to keep my eye on. Be Very Specific in the recommendations. "))    
    chatbot({"nodes": nodes, "relationships": relationships})