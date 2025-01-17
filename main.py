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

# Initialize Ollama LLM
llm = Ollama(model="llama3.3:70b-instruct-q4_K_M", base_url="http://localhost:11434")

# Initialize Graph Transformer
llm_transformer = LLMGraphTransformer(llm=llm)

def process_text_to_graph(user_info: str, text: str, chunk_size: int = 800, chunk_overlap: int = 200):
    """
    Process a long text into a knowledge graph by splitting it into chunks
    and combining the results.
    """
    # Create text splitter
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
            combined_text = user_info + "\n" + doc.page_content
            combined_document = Document(page_content=combined_text)
            graph_doc = llm_transformer.convert_to_graph_documents([combined_document])[0]
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

    prompt = f"""
    Hello, I’m your personal recommendation agent, and my sole job is to provide you with the best possible suggestions based on the information you’ve provided. I analyze the data meticulously to ensure that I deliver results tailored to your preferences and needs. I take pride in offering recommendations that are not only accurate but also thoughtful and relevant to your goals.

    Here’s what I’ve analyzed from the knowledge graph I’m working with:
    {prompt}
    
    I’ll now generate the best possible recommendations for you. Please give me a moment.
    """

    print("the promt:\n", prompt)
    # Generate recommendation using the Llama model
    recommendation = llm.invoke(prompt)
    return recommendation

# Example usage
if __name__ == "__main__":
    user_info = """ Basic Information

    Name: Alex Morgan
    Age: 32
    Gender: Non-binary
    Location: Seattle, Washington, USA
    Profession: Data Scientist at a FinTech startup
    Education: Master's in Computer Science from the University of Washington
    Relationship Status: Single"""
    text1 = """

    Preferences
    Food 🍴

    Favorite Cuisines:
        Japanese (especially sushi and ramen)
        Italian (authentic pasta dishes)
        Ethiopian (loves injera and stews)
    Dietary Preferences:
        Mostly pescatarian but enjoys trying new food trends.
    Favorite Restaurants in Seattle:
        Maneki (Japanese)
        The Pink Door (Italian)
        Tilikum Place Café (Brunch and Dutch pancakes)

    Books 📚

    Favorite Genres:
        Science Fiction: Dune by Frank Herbert
        Non-Fiction: Sapiens by Yuval Noah Harari
        Modern Classics: The Catcher in the Rye by J.D. Salinger
    Current Reading List:
        The Midnight Library by Matt Haig
        Atomic Habits by James Clear
    Preferred Format: Audiobooks during commutes and hardcover for evenings.

    Investments 💰

    Investment Preferences:
        Cryptocurrency: Holds Bitcoin (BTC), Ethereum (ETH), and Cardano (ADA).
        Index Funds: Invests in S&P 500 ETFs like VOO.
        Tech Stocks: Holds shares in Tesla, Google, and Amazon.
    Risk Tolerance: Moderate, enjoys exploring new technologies but balances with safe investments.

    Twitter Following/Follower 🐦

    Followers: 2,345
    Following: 1,892
    Types of Accounts Followed:
        Data Science influencers (e.g., @AndrewYNg, @TDataScience)
        Crypto enthusiasts (e.g., @VitalikButerin, @CoinDesk)
        Movie critics and indie filmmakers.
        Food bloggers and Seattle-based restaurant reviewers.
    Common Tweets:
        Sharing Python/ML tips.
        Commenting on cryptocurrency trends.
        Reviewing sci-fi movies.

    Hobbies and Interests 🌟

    Tech Interests:
        Building side projects in machine learning and blockchain.
        Learning Rust and exploring the RISC-V architecture.
    Leisure Activities:
        Hiking and photography around the Pacific Northwest.
        Gaming: Favorite titles include The Witcher 3 and Cyberpunk 2077.
        Watching anime: Favorites include Attack on Titan and Steins;Gate.

    Music Preferences 🎵

    Favorite Genres:
        Lo-fi beats for work.
        Synthwave and electronic for workouts.
        Indie rock (e.g., The 1975, Tame Impala).

    Fitness and Wellness 🏋️‍♂️

    Routine:
        Morning yoga and meditation.
        Runs 5k twice a week, aiming for a half-marathon.
    Apps Used:
        Strava for running.
        Headspace for meditation.
    """
    # Process the text and generate the graph
    nodes, relationships = process_text_to_graph(user_info=user_info, text=text1)
    
    # Create and save the visualization
    create_interactive_graph(nodes, relationships)
    print("\n response: \n",generate_recommendation({"nodes": nodes, "relationships": relationships}, "give me recommendation of which movies i could watch "))