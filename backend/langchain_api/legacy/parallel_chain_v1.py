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
    input_variables=["situation"],
    template="""Extract the key facts from the situation description. Be very concise.
    
    Current situation: {situation}
    
    You must respond with valid JSON only, in exactly this format:
    {{"1": "fact 1", "2": "fact 2", "3": "fact 3"}}
    
    Don't write ```json or anything else, just the JSON object.
    ---
    Example:
    User: I am a student living in Berlin. I want to move to a bigger apartment that is close to my uni.
    Response: {{"1": "student", "2": "Berlin", "3": "bigger apartment"}}
    ---
    """
)

# Create chains
"""
situation -> parallel(generate possible lifestyles, extract key facts) -> generate long lifestyles
"""
lifestyle_chain = lifestyle_generator_prompt | model
extraction_chain = extraction_prompt | model
parallel_chain = RunnableParallel(
    lifestyles=lifestyle_chain,
    facts=extraction_chain
)

# State memory
memory = ChatMessageHistory()

def generate_lifestyles(situation: str):
    results = parallel_chain.invoke({"situation": situation})
    
    print("\nLifestyles: ", results["lifestyles"])
    print("Facts: ", results["facts"])

    # Parse the string results into JSON
    try:
        return json.loads(results["lifestyles"]), json.loads(results["facts"])
    except json.JSONDecodeError:
        return {"error": "Failed to parse response into JSON"}

user_input = input("Please describe your current situation and future plans: ")
if not user_input:
    default_input = "I am a student living in Berlin. I want to move to a bigger apartment that is close to my uni."
    print(f"No input provided. Using default input: {default_input}")
    user_input = default_input

print(generate_lifestyles(user_input))
