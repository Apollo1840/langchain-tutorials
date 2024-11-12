from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
google_api_key = os.getenv("GOOGLE_API_KEY", "YourAPIKey")

# print(openai_api_key)
# print(google_api_key)

from langchain import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document

prompt = PromptTemplate(input_variables=["text"], template="I am a new bee, please anwser me {text}")

text = prompt.format(text="how to use LangChain")
print(type(text), text)
print()

chat_msg = [
    SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
    HumanMessage(content="I like the beaches where should I go?"),
    AIMessage(content="You should go to Nice, France"),
    HumanMessage(content="What else should I do when I'm there?")
]
print(type(chat_msg[0]), chat_msg[0].content)
print()

document = Document(page_content="This is my document. It is full of text that I've gathered from other places",
                    metadata={
                        'my_document_id': 234234,
                        'my_document_source': "The LangChain Papers",
                        'my_document_create_time': 1680013019
                    })
print(type(document), document.page_content)
print()

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

chat = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=openai_api_key)
# chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0, google_api_key=google_api_key)

ai_message = chat(chat_msg)
print(ai_message)
print()
print(ai_message.content)

from langchain.chains import LLMChain, SimpleSequentialChain

template = """Your job is to come up with a classic dish from the area that the users suggests, and only respond the short dish name
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Holds my 'location' chain
location_chain = LLMChain(llm=chat, prompt=prompt_template)

location_chain.run("China")

template = """Given a meal, give a short and simple recipe on how to make that dish at home. please respond in a very structured way.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=chat, prompt=prompt_template)

print(meal_chain.run("Peking Duck"))

overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
print(overall_chain.run("Rome"))


# function call output


def get_current_weather(location, unit):
    return 37.5

functions=[{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    }
}
]

output = chat(messages=[
    SystemMessage(content="You are an helpful AI bot"),
    HumanMessage(content="What’s the weather like in Boston right now?")
], functions=functions)

f_call_str = output.additional_kwargs["function_call"]
print(f_call_str)

f_call = "{func_name}(**{func_kwargs})".format(func_name=f_call_str["name"], func_kwargs=f_call_str["arguments"])
print(f_call)
eval(f_call)

