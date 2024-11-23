from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAI
import gemini_utils

# Define the chatbot logic
class LifePlanningChatbot:
    def __init__(self, model, key_information_extractor):
        self.model = model
        self.key_information_extractor = key_information_extractor
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.user_state = {
            "preferences": {},
            "enough_information": False,
        }

        self.prompt_template = PromptTemplate(
            input_variables=["history", "user_input", "state"],
            template="""
            You are a life-planning assistant. Help the user clarify their life goals for the next 10 years, guiding them on aspects like:
            - Neighborhood preferences
            - Proximity to work, schools, or kindergartens
            - Cultural, social, and food preferences
            - Pet-friendliness and nearby supermarkets

            Always update their preferences based on the conversation, and decide when you have enough information to summarize their goals.

            Current user preferences (state):
            {state}

            Chat History:
            {history}

            User: {user_input}

            Reply with a detailed and empathetic response.
            """
        )

        self.chat_chain = ConversationChain(
            llm=VertexAI("gemini-1.5-flash"),
            memory=self.memory,
            prompt=self.prompt_template,
            verbose=True,
        )

    def update_user_state(self, user_input):
        # Extract key information using the model
        extracted_info = self.key_information_extractor.extract(user_input)
        for key, value in extracted_info.items():
            if key not in self.user_state["preferences"]:
                self.user_state["preferences"][key] = value
            else:
                # Update values (e.g., merge or prioritize latest input)
                self.user_state["preferences"][key] = value

        # Decide if we have enough information to move forward
        self.user_state["enough_information"] = self.check_enough_information()

    def check_enough_information(self):
        # Example: Require a few key fields to be filled
        required_keys = ["work_proximity", "school_proximity", "cultural_preferences"]
        return all(key in self.user_state["preferences"] for key in required_keys)

    def run(self):
        print("Welcome to the Life Planning Chatbot! Let's plan your next 10 years.")
        while not self.user_state["enough_information"]:
            user_input = input("You: ")
            self.update_user_state(user_input)
            state_summary = f"Preferences: {self.user_state['preferences']}"
            response = self.chat_chain.predict(
                user_input=user_input, state=state_summary
            )
            print(f"Chatbot: {response}")

        print("\nWe have enough information! Here's your summary:")
        print(self.user_state["preferences"])


# Define the key information extractor
class KeyInformationExtractor:
    def __init__(self, model):
        self.model = model

    def extract(self, text):
        # A simple extraction logic, this could use regex or fine-tuned ML logic
        extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract key preferences from the user's input. Look for:
            - Proximity to work
            - Proximity to schools/kindergartens
            - Cultural and social preferences
            - Pet-related needs
            - Food and supermarket preferences

            User input:
            {text}

            Extracted preferences:
            """
        )

        # Use the generative model to extract structured information
        response = self.model.generate(
            input_text=extraction_prompt.render({"text": text}),
            max_output_tokens=100,
        )
        # Assume response is a JSON-like dict structure
        return eval(response.text)


# Initialize the script
if __name__ == "__main__":
    # Load the Gemini model
    model = gemini_utils.init_model()

    # Initialize the chatbot and extractor
    extractor = KeyInformationExtractor(model)
    chatbot = LifePlanningChatbot(model, extractor)

    # Run the chatbot
    chatbot.run()
