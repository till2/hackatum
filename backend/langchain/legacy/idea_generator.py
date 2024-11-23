from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAI
import vertexai
import json
from langchain_community.chat_message_histories import ChatMessageHistory

# Initialize Vertex AI model
vertexai.init(project="hackathum24mun-28", location="europe-west3")
model = VertexAI(model_name="gemini-1.5-flash")

# Create prompt templates
lifestyle_generator_prompt = PromptTemplate(
    input_variables=["situation"],
    template="""Based on this person's situation, generate 3 different possible lifestyles they might have in 10 years. 
    Each lifestyle should be a brief list of keywords or short phrases.
    
    Current situation: {situation}
    
    Provide the output as a JSON object where keys are 1, 2, 3 and values are the keywords as text.
    Example format: {{"1": "remote work, 2 kids, yoga instructor", "2": "digital nomad, single, travel blogger", "3": "startup founder, married, city life"}}"""
)

story_generator_prompt = PromptTemplate(
    input_variables=["lifestyles"],
    template="""Convert these lifestyle keywords into three coherent narrative paragraphs.
    
    Lifestyles: {lifestyles}
    
    Provide the output as a JSON object where keys are 1, 2, 3 and values are the full narrative paragraphs."""
)

fact_extractor_prompt = PromptTemplate(
    input_variables=["situation"],
    template="""Extract the key facts from this person's current situation. 
    List them as concise, separate points.
    
    Situation: {situation}"""
)

# Create chains using the new RunnableSequence syntax
lifestyle_chain = lifestyle_generator_prompt | model
story_chain = story_generator_prompt | model
fact_chain = fact_extractor_prompt | model

# Create memory with the new syntax (see migration guide)
memory = ChatMessageHistory()

# Combined chain
overall_chain = (
    {"situation": lambda x: x["situation"]} 
    | lifestyle_chain 
    | {"lifestyles": lambda x: x} 
    | story_chain 
    | {"stories": lambda x: x, "situation": lambda x: x["situation"]} 
    | fact_chain
)

def generate_futures(situation: str):
    result = overall_chain.invoke({
        "situation": situation
    })
    
    # Store facts in memory
    memory.save_context({"input": situation}, {"output": result["facts"]})
    
    return {
        "lifestyles": json.loads(result["lifestyles"]),
        "stories": json.loads(result["stories"]),
        "facts": result["facts"]
    }

user_input = input("Please describe your current situation and future plans: ")
if not user_input:
    default_input = "I am a student living in Berlin. I want to move to a bigger apartment that is close to my uni."
    print(f"No input provided. Using default input: {default_input}")
    user_input = default_input

print(generate_futures(user_input))