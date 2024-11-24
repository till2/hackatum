from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAI
import vertexai
import json
from langchain_community.chat_message_histories import ChatMessageHistory

vertexai.init(project="hackathum24mun-28", location="europe-west3")
lifestyle_model = VertexAI(model_name="gemini-1.5-flash")

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

# Create chains
lifestyle_chain = lifestyle_generator_prompt | lifestyle_model

# State memory
memory = ChatMessageHistory()


def generate_lifestyles(situation: str):
    result = lifestyle_chain.invoke({"situation": situation})
    
    print("Result: ", result)

    # Parse the string result into JSON
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"error": "Failed to parse response into JSON"}

user_input = input("Please describe your current situation and future plans: ")
if not user_input:
    default_input = "I am a student living in Berlin. I want to move to a bigger apartment that is close to my uni."
    print(f"No input provided. Using default input: {default_input}")
    user_input = default_input

print(generate_lifestyles(user_input))
