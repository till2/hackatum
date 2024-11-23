from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from vertexai.generative_models import Part, Content, GenerationConfig
from gemini_utils import init_model

# Initialize the model
model = init_model()

# Generation configuration
generation_config = GenerationConfig(
    max_output_tokens=1024,
    temperature=0.5,
    top_p=0.95,
)

# Define Prompt Templates
life_planning_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template="""You are a professional life planning advisor helping people make informed decisions about their future housing choices. 
Your role is to guide them through a comprehensive discussion about their life plans for the next 10 years, helping them uncover their true needs and preferences beyond their initial assumptions.

During the conversation, get the user to answer the following two questions:
1. What are your current living situation and immediate housing needs (e.g. moving to a new city, more space, short trip to work, etc.)?
2. Where do you see yourself professionally and personally in the next 5-10 years?

Start with the first question and then once it is sufficiently answered, move on to the second question.

After these initial questions, engage in a natural conversation that explores various life aspects relevant to housing choices, such as:
- Career growth and workplace location
- Family planning (marriage, children, pets)
- Lifestyle preferences (urban/suburban, social life, cultural activities)
- Daily necessities (transportation, shopping, healthcare)
- Educational needs (schools, universities)
- Work-life balance
- Budget and financial planning

Previous conversation:
{chat_history}

---- Additional instructions ----

Your tone:
Respond professionally and empathetically. 

Your task:
Help users discover what they truly need, not just what they think they want.
Also, make sure to ask follow-up questions to get more information. 

Your question length:
Always write concise questions and don't overwhelm the user with long questions or answers.
BE CONCISE and to the point, but also empathetic and friendly.

Start by asking the first question (1. What are your current living situation and immediate housing needs (e.g. moving to a new city, more space, short trip to work, etc.)?).
Then let the user answer and respond with follow-up questions later on. This is not a one-shot conversation, so don't try to output the user answer in your response.
"""
)

preference_extractor_prompt = PromptTemplate(
    input_variables=["conversation"],
    template="""Based on the following conversation, extract and summarize the key preferences and requirements for housing:

{conversation}

Please provide a concise summary of:
1. Must-have features
2. Location preferences
3. Lifestyle requirements
4. Budget constraints
5. Long-term considerations

Format the response as a clear, structured summary."""
)

class State(TypedDict):
    messages: Annotated[List, add_messages]
    key_preferences: Optional[str]

graph_builder = StateGraph(State)

memory = MemorySaver()

# Define the chatbot node
def chatbot(state: State):
    # Convert each message to its content
    chat_history = "\n".join([msg.content for msg in state["messages"]])
    prompt = life_planning_prompt.format(chat_history=chat_history)
    response = model.generate_content(
        contents=[Content(role="user", parts=[Part.from_text(prompt)])],
        generation_config=generation_config
    ).text
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)

# Define the preference extractor node
def extractor(state: State):
    # Convert each message to its content
    conversation = "\n".join([msg.content for msg in state["messages"]])
    summary = model.generate_content(
        contents=[Content(role="user", parts=[Part.from_text(preference_extractor_prompt.format(conversation=conversation))])],
        generation_config=generation_config
    ).text
    return {"key_preferences": summary, "messages": [summary]}

graph_builder.add_node("extractor", extractor)

# Define conditional logic to determine when to extract preferences
def check_completion(state: State):
    # Define your logic to decide when enough information is gathered
    # For example, after a certain number of messages or specific keywords
    if len(state["messages"]) >= 10 or ("summary" in state["messages"][-1].content.lower()):
        return "extractor"
    return "chatbot"

graph_builder.add_conditional_edges(
    "chatbot",
    check_completion,
    {"extractor": "extractor", "chatbot": "chatbot", END: END},
)

# Define edges
graph_builder.add_edge("extractor", END)
graph_builder.add_edge(START, "chatbot")

# Compile the graph with checkpointing
graph = graph_builder.compile(checkpointer=memory)

# Main execution flow
def main():
    print("Welcome to your Life Planning Advisor!")
    print("Let's discuss your future housing needs and life plans.")
    print("(Type 'exit', 'quit', 'done', or 'next' when you're ready to view your preferences summary)")
    
    # Initialize the conversation
    events = graph.stream({"messages": []}, {"thread_id": "life_planner"})
    
    # Process all initial events (not just the first one)
    for event in events:
        if "messages" in event:
            for message in event["messages"]:
                print("\nAdvisor:", message)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'done', 'next']:
            # Trigger to extract preferences if user wants to end the conversation
            snapshot = graph.get_state({"thread_id": "life_planner"})
            preferences = snapshot.values.get("key_preferences", "No preferences extracted.")
            print("\nBased on our conversation, here's a summary of your preferences and requirements:")
            print(preferences)
            break
        
        # Stream user input to the graph
        events = graph.stream({"messages": [("user", user_input)]}, {"thread_id": "life_planner"})
        for event in events:
            if "messages" in event:
                print("\nAdvisor:", event["messages"][-1])
        
        # Check if the graph has reached the extractor node
        snapshot = graph.get_state({"thread_id": "life_planner"})
        if snapshot.next == ("extractor",):
            preferences = snapshot.values.get("key_preferences", "No preferences extracted.")
            print("\nBased on our conversation, here's a summary of your preferences and requirements:")
            print(preferences)
            break

if __name__ == "__main__":
    main()
