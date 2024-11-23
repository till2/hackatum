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
    {{"1": "lifestyle 1 description", "2": "lifestyle 2 description", "3": "lifestyle 3 description"}}
    
    Don't write ```json or anything else, just the JSON object.
    ---
    Example:
    User: I am a student living in Berlin. I want to move to a bigger apartment that is close to my uni.
    Response: {{"1": "remote work, 2 kids, yoga instructor", "2": "digital nomad, single, travel blogger", "3": "startup founder, married, city life"}}
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
    You can infer the most likely facts.
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
    Lifestyles: {{"1": "remote work, 2 kids, yoga instructor", "2": "digital nomad, single, travel blogger", "3": "startup founder, married, city life"}}
    Response: {{"1": {{"1": "💻", "2": "👶", "3": "🧘"}}, "2": {{"1": "🌎", "2": "📝"}}, "3": {{"1": "💼", "2": "💑", "3": "🌆"}}}}
    ---
    """
)

# Create chains
"""
situation -> parallel(update key facts, generate possible lifestyles) -> generate emojis for lifestyles
"""
extraction_chain = extraction_prompt | model
lifestyle_chain = lifestyle_generator_prompt | model
emoji_chain = emoji_prompt | model

# State memory
memory = ChatMessageHistory()

def get_previous_facts():
    messages = memory.messages
    if not messages:
        return "{}"
    
    # Get the last AI message which contains facts
    for message in reversed(messages):
        if message.type == "ai":
            return message.content
    return "{}"

def generate_lifestyles(situation: str):
    previous_facts = get_previous_facts()
    
    # Create parallel chain with extraction getting previous facts
    parallel_chain = RunnableParallel(
        lifestyles=lifestyle_chain,
        facts={
            "previous_facts": lambda x: x["previous_facts"],
            "situation": lambda x: x["situation"], 
        } | extraction_chain
    )
    
    results = parallel_chain.invoke({
        "previous_facts": previous_facts,
        "situation": situation,
    })
    
    # Generate emojis based on the lifestyles
    emoji_results = emoji_chain.invoke({"lifestyles": results["lifestyles"]})
    
    print("--------------- RESULTS ---------------")
    print("Facts: ", results["facts"])
    print("Lifestyles: ", results["lifestyles"])
    print("Emojis: ", emoji_results)
    
    # Store the interaction in memory
    memory.add_user_message(situation)
    memory.add_ai_message(results["facts"])

    # Parse the string results into JSON
    try:
        lifestyles = json.loads(results["lifestyles"])
        facts = json.loads(results["facts"])
        emojis = json.loads(emoji_results)
        return lifestyles, facts, emojis
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

            lifestyles, facts, emojis = generate_lifestyles(user_input)
            
            print("--------------- JSON -----------------")
            print("Facts: ", facts)
            print("Lifestyles: ", lifestyles)
            print("Emojis: ", emojis)
            
        except KeyboardInterrupt:
            print("\nExiting program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()