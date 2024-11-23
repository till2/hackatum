from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnableParallel
from langchain_community.chat_message_histories import ChatMessageHistory
import vertexai
import json

vertexai.init(project="hackathum24mun-28", location="europe-west3")
model = VertexAI(model_name="gemini-1.5-flash")

# Modify prompt template to be more explicit about JSON structure
lifestyle_generator_prompt = PromptTemplate(
    input_variables=["situation"],
    template="""Based on this person's situation, generate 3 different possible lifestyles they might have in 10 years. 
    Each lifestyle should be a brief list of keywords or short phrases.
    
    Current situation: {situation}
    
    You must respond with valid JSON only, in exactly this format:
    {{"1": {{"1": "aspect 1", "2": "aspect 2", "3": "aspect 3", "4": "aspect 4", "5": "aspect 5"}},
     "2": {{"1": "aspect 1", "2": "aspect 2", "3": "aspect 3", "4": "aspect 4"}},
     "3": {{"1": "aspect 1", "2": "aspect 2", "3": "aspect 3"}}}}
     
    Don't write ```json or anything else, just the JSON object.
    ---
    Example:
    User: I am a student living in Berlin. I want to move to a bigger apartment that is close to my uni.
    Response: {{"1": {{"1": "remote work", "2": "2 kids", "3": "yoga teacher", "4": "suburbs", "5": "consultant"}}, 
               "2": {{"1": "digital nomad", "2": "single", "3": "travel blogger", "4": "crypto trader"}}, 
               "3": {{"1": "startup founder", "2": "married", "3": "city life"}}}}
    ---
    """
)

extraction_prompt = PromptTemplate(
    input_variables=["situation", "previous_facts"],
    template="""Extract and update key facts from the situation description. Consider previous facts and add/update with new information.

    Previous facts: {previous_facts}
    Current situation: {situation}
    
    You must respond with valid JSON only, in exactly this format:
    {{"1": "fact 1", "2": "fact 2", "3": "fact 3"}}
    
    Combine previous and new facts, prioritizing the most relevant information.
    If the old facts don't make sense anymore given the new situation, remove them.
    If they are still relevant (e.g. where they live etc.), keep or update them depending on the new situation. 
    You can infer the most likely facts. Also if the user inputs irrelevant information, you can ignore it.
    But don't make up completely new facts that are not mentioned in the situation.
    Don't write ```json or anything else, just the JSON object.
    ---
    Example:
    Previous facts: {{"1": "student", "2": "lives alone", "3": "works part-time"}}
    User: I am a student living in Berlin. I want to move to a bigger apartment that is close to my uni.
    Response: {{"1": "student in Berlin", "2": "seeking bigger apartment", "3": "wants to be close to uni"}}
    ---
    """
)

fact_classifier_prompt = PromptTemplate(
    input_variables=["facts"],
    template="""Classify each fact into one of these predefined categories: "education", "family", "work", "hobbies", "lifestyle", "other".
    Choose the most appropriate category for each fact based on these guidelines:
    
    Only classify facts that are mentioned in the user's situation. If a fact is not mentioned in the situation, it should be classified as "other".
    
    - education: Facts about schooling, university, studying
    - family: Facts about relationships, children, parents, significant other
    - work: Facts about jobs and career
    - hobbies: Facts about recreational activities (like sports)
    - lifestyle: Facts about living situation, daily habits, preferences (like bars, restaurants, museums, etc.)
    - other: Facts that don't clearly fit the above categories (if you are unsure, rather classify as "other")

    If a fact could fit multiple categories, choose the one that is most central to the fact's meaning.
    For example: "wants to study close to family" should be classified as "family" since the main focus is on family.

    Facts: {facts}
    
    You must respond with valid JSON only, in exactly this format:
    {{"1": "education", "2": "lifestyle", "3": "work"}}
    
    Constraints:
    - Each fact must be classified into exactly one category
    - The value must be exactly one of: "education", "family", "work", "hobbies", "lifestyle", "other"
    - The key must be the fact number (index) from the input
    - The value must be a lowercase string
    - The keys must match the fact numbers from the input
    - Do not add explanations or classifications for facts that don't exist in the input
    - Multiple facts can be classified into the same category
    
    Don't write ```json or anything else, just the JSON object.
    ---
    Example:
    Facts: {{"1": "student in Berlin", "2": "seeking bigger apartment", "3": "wants to be close to uni", "4": "likes to travel", "5": "works at Google", "6": "likes dogs"}}
    Response: {{"1": "education", "2": "lifestyle", "3": "education", "4": "lifestyle", "5": "work", "6": "other"}}
    ---
    """
)

housing_facts_prompt = PromptTemplate(
    input_variables=["facts", "previous_housing_facts"],
    template="""Based on the general facts, extract and categorize location-relevant information for finding a suitable home.
    Consider previous housing facts and update with new information.

    Previous housing facts: {previous_housing_facts}
    Current facts: {facts}
    
    You must respond with valid JSON only, in exactly this format:
    {{
        "education": ["TUM Garching Campus"],
        "family": ["Dresdner Straße Munich"],
        "work": ["Google Munich Safety Engineering Center"],
        "hobbies": ["Climbing Factory"],
        "lifestyle": ["Museum", "Art Gallery"]
    }}
    
    If you know any specific details, e.g. the location where the grandparents live, add it to the corresponding category
    in a way that we can input it into a valid google maps search (here, Dresdner Straße Munich is the grandparents' address).
    As a bad example, "grandparents" is not a valid location. Also "climbing" is valid because google maps 
    will return climbing gyms. But "cycling" is not valid because google maps will likely just return cycling stores.
    => You can judge what makes a valid location by what you would search on google maps.
            
    Rules:
    - Each category should contain specific, searchable locations
    - Work locations should be very specific (exact company location)
    - Only include facts that are relevant for housing search
    - Lists can be empty if no relevant information is available
    - Ignore facts that don't fit into any category
    - If you are unsure about a fact, rather leave it out than guessing
    - Make locations as specific as possible while staying truthful to the facts
    - Don't write generic entries like "company XYZ" or "school XYZ"
    - If you don't know the exact location just write "university" or leave it empty if it wouldn't return useful map search results
    Don't write ```json or anything else, just the JSON object.
    """
)

emoji_prompt = PromptTemplate(
    input_variables=["lifestyles"],
    template="""Based on these three possible lifestyles, generate 1-3 distinct emojis for each lifestyle that best represent it visually.
    Use different emojis for each lifestyle - don't repeat emojis across lifestyles.
    
    Lifestyles: {lifestyles}
    
    You must respond with valid JSON only, in exactly this format:
    {{"1": {{"1": "emoji1", "2": "emoji2", "3": "emoji3"}}, 
      "2": {{"1": "emoji1", "2": "emoji2"}},
      "3": {{"1": "emoji1", "2": "emoji2", "3": "emoji3"}}}}
    
    Use only actual emojis, no text. Use 1-3 emojis per lifestyle.
    You can select how many emojis you need (1-3) to represent the lifestyle well in a visual way.
    Use AT LEAST ONE emoji per lifestyle. DO NOT use more than 3 emojis per lifestyle. 
    For each index in a lifestyle, exactly write one emoji.
    If there is more information than 3 emojis can covey, you have to select the 3 most important ones.
    Don't write ```json or anything else, just the JSON object.
    ---
    Example:
    Lifestyles: {{"1": {{"1": "remote work", "2": "parent of 2 kids", "3": "yoga instructor", "4": "suburban house", "5": "part-time consultant"}},
                 "2": {{"1": "digital nomad", "2": "single", "3": "travel blogger", "4": "cryptocurrency investor"}},
                 "3": {{"1": "startup founder", "2": "married", "3": "city life"}}}}
    Response: {{"1": {{"1": "💻", "2": "👶", "3": "🏡"}}, "2": {{"1": "🌎", "2": "💻", "3": "📝"}}, "3": {{"1": "💼", "2": "💑", "3": "🌆"}}}}
    ---
    """
)

# Create chains
extraction_chain = extraction_prompt | model
lifestyle_chain = lifestyle_generator_prompt | model
emoji_chain = emoji_prompt | model
housing_facts_chain = housing_facts_prompt | model
fact_classifier_chain = fact_classifier_prompt | model

# State memory
memory = ChatMessageHistory()
housing_memory = ChatMessageHistory()

def get_previous_facts():
    messages = memory.messages
    if not messages:
        return "{}"
    for message in reversed(messages):
        if message.type == "ai":
            return message.content
    return "{}"

def get_previous_housing_facts():
    messages = housing_memory.messages
    if not messages:
        return "{}"
    for message in reversed(messages):
        if message.type == "ai":
            return message.content
    return "{}"

def generate_lifestyles(situation: str):
    previous_facts = get_previous_facts()
    previous_housing_facts = get_previous_housing_facts()
    
    # First parallel chain for facts and lifestyles
    first_parallel = RunnableParallel(
        lifestyles=lifestyle_chain,
        facts={
            "previous_facts": lambda x: x["previous_facts"],
            "situation": lambda x: x["situation"], 
        } | extraction_chain
    )
    
    first_results = first_parallel.invoke({
        "previous_facts": previous_facts,
        "situation": situation,
    })
    
    # Second parallel chain for emojis and housing facts
    second_parallel = RunnableParallel(
        emojis={"lifestyles": lambda x: x["lifestyles"]} | emoji_chain,
        housing_facts={
            "facts": lambda x: x["facts"],
            "previous_housing_facts": lambda x: x["previous_housing_facts"]
        } | housing_facts_chain,
        fact_categories={"facts": lambda x: x["facts"]} | fact_classifier_chain
    )
    
    second_results = second_parallel.invoke({
        "lifestyles": first_results["lifestyles"],
        "facts": first_results["facts"],
        "previous_housing_facts": previous_housing_facts
    })
    
    print("--------------- RESULTS ---------------")
    print("Facts: ", first_results["facts"])
    print("Fact Categories: ", second_results["fact_categories"])
    print("Lifestyles: ", first_results["lifestyles"])
    print("Emojis: ", second_results["emojis"])
    print("Housing Facts: ", second_results["housing_facts"])
    
    # Store the interactions in memory
    memory.add_user_message(situation)
    memory.add_ai_message(first_results["facts"])
    housing_memory.add_user_message(first_results["facts"])
    housing_memory.add_ai_message(second_results["housing_facts"])

    # Parse the string results into JSON
    try:
        lifestyles = json.loads(first_results["lifestyles"])
        facts = json.loads(first_results["facts"])
        fact_categories = json.loads(second_results["fact_categories"])
        emojis = json.loads(second_results["emojis"])
        housing_facts = json.loads(second_results["housing_facts"])
        return lifestyles, facts, fact_categories, emojis, housing_facts
    except json.JSONDecodeError:
        return {"error": "Failed to parse response into JSON"}

def main():
    while True:
        try:
            user_input = input("\nPlease describe your current situation and future plans: ")
            if user_input == "1":
                default_input_1 = "I am a student living in Potsdam. I want to move to a bigger apartment that is close to my uni."
                print(f"Using default input 1: {default_input_1}")
                user_input = default_input_1
            elif user_input == "2":
                default_input_2 = "I got a job now. I want to move to a bigger apartment that is close to my work in Berlin."
                print(f"Using default input 2: {default_input_2}")
                user_input = default_input_2

            lifestyles, facts, fact_categories, emojis, housing_facts = generate_lifestyles(user_input)
            
            print("--------------- JSON -----------------")
            print("Facts: ", facts)
            print("Fact Categories: ", fact_categories)
            print("Lifestyles: ", lifestyles)
            print("Emojis: ", emojis)
            print("Housing Facts: ", housing_facts)
            
        except KeyboardInterrupt:
            print("\nExiting program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()