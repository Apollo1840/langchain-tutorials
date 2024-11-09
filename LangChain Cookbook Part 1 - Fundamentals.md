# LangChain Cookbook

## Components

### Schema

- Text, Prompt
```python
  prompt = PromptTemplate(input_variables=["text"], template="I am a new bee, please anwser me {text}")
  text = prompt.format("how to use LangChain")
```

- chatMessages: 
```python
[
 SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
 HumanMessage(content="I like the beaches where should I go?"),
 AIMessage(content="You should go to Nice, France"),
 HumanMessage(content="What else should I do when I'm there?")
]
```
- Document:
```python
Document(page_content="This is my document. It is full of text that I've gathered from other places",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "The LangChain Papers",
             'my_document_create_time' : 1680013019
         })
```

### Models

| Models                | Example                               | Input            | Output       |
|----------------------|---------------------------------------|------------------|--------------|
| **Language model**   | `OpenAI(model_name, openai_api_key)`  | text            | text         |
| **Chat model**       | `ChatOpenAI(temperature, openai_api_key)` | chatMessages  | a chatMessage  |
| **TextEmbedding model** | `OpenAIEmbeddings(openai_api_key)` | text            | vector       |


### Chain

Use `LLMChain` to create a **Chain Component**.

Chain 1:

```python

template = """Your job is to come up with a classic dish from the area that the users suggests, and only respond the short dish name
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Holds my 'location' chain
location_chain = LLMChain(llm=chat, prompt=prompt_template)



```

Chain 2:

```python

template = """Given a meal, give a short and simple recipe on how to make that dish at home. please respond in a very structured way.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=chat, prompt=prompt_template)


```

Link them as a LONG CHAIN:

```python

overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
print(overall_chain.run("Rome"))

```


### Tools

| Tools            | Example                                      | Usage             | Input | Output                                          |
|------------------|----------------------------------------------|-------------------|-------|-------------------------------------------------|
| **SearchEngine** | `search = TavilySearchResults()`             | `search.invoke()` | text  | list of web search results as `{"url":"", "content:""}` |
| **Retriever**    | `retriever = FAISS.from_documents().as_retriever()` | `retriever.invoke()` | text  | list of `Documents`                             |


### Agents
- **ToolAgent** (eg. `agent = create_tool_calling_agent(llm, tools, prompt)`)
  usage: 
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke(), it will think with llm and decide whether and how to use the tools.

## Capablities


### Basic

```python

chat = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=openai_api_key)
# chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0, google_api_key=google_api_key)

chat_msg = [
 SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
 HumanMessage(content="I like the beaches where should I go?"),
 AIMessage(content="You should go to Nice, France"),
 HumanMessage(content="What else should I do when I'm there?")
]

ai_message = chat(chat_msg)
print(ai_message)
print()
print(ai_message.content)

```

### Augment the Input - PromptEngineering - Semantic Examples

#### Target
Suppose we want to ask the LLM to 

```python
text = """
    Give the location an item is usually found in
    
    Input: plant
    Output:
    
    """
```

In order to guide the model to output a more desirable answer, 
we can append some examples between the context and question form.

eg.

```python

text = """
    Give the location an item is usually found in
    
    <examples>
    
    Input: plant
    Output:
    
    """
```

i.e.

```python
text = """
    Give the location an item is usually found in
    
    Example Input: tree
    Example Output: ground
    
    Example Input: bird
    Example Output: nest
    
    Input: plant
    Output:

"""
```

And we can actually prepare a lots of examples but we only want the program to choose examples which are semantically similar to the original request.

#### Implementation

OK, so we have:

```python
examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]
```

We will use a semantic selector based on similarity:

```python
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples, 
    
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(openai_api_key=openai_api_key), 
    
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma, 
    
    # This is the number of examples to produce.
    k=2
)
```

We also need a example_prompt to change one example `{"input": "pirate", "output": "ship"}` to one example text like:
`"Example Input: pirate\nExample Output: ship"`:


```python
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)
```

Finally, we can use `FewShotPromptTemplate` to implement this in an eligent way.

```python
similar_prompt = FewShotPromptTemplate(
    # Customizations that will be added to the top and bottom of your prompt
    prefix="Give the location an item is usually found in",
    suffix="Input: {input}\nOutput:",
    
    # What inputs your prompt will receive
    input_variables=["input"],
    
    # Your prompt
    example_prompt=example_prompt,
    
    # The object that will help select examples
    example_selector=example_selector,
)

# Select a noun!
my_noun = "plant"
# my_noun = "student"

print(similar_prompt.format(input=my_noun))

```

And we can send the result as the input to a LLM.

(p.s no garantee that the output is one word)



### Format the Output - Function Call

#### Target
Suppose we want to the LLM to output a function call like
```python

"""
    {
        'name': 'get_current_weather', 
        'arguments': '{
            "location": "Boston, MA"\n}'
    }
"""

```

instead of normal text.



#### Implementation

You just need to specify the Function API clearly to a **chat model**.

```python

chat = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=1, openai_api_key=openai_api_key)

output = chat(messages=
     [
         SystemMessage(content="You are an helpful AI bot"),
         HumanMessage(content="What’s the weather like in Boston right now?")
     ],
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
)

print(output["additional_kwargs"]["function_call"])
```

### Format the Output - JSON

#### Target

So we want the LLM to output a dictionary object like this:

```json

{
  "english": "hello world",
  "german": "hallo Welt"
}

```

In this case, we will input a sentence in english or german, and we expect the model to output a dictionary as above.

#### Implementation

Just like what we did in PromptEngineering, we will insert a snippet of sentences into the Prompt.

The instruction snippet is look like:

```python

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
  
    ```json
    {
        "english": string  // This a sentence in english.
        "german": string  // This e sentence in german.
    }
    ```

```

Luckily, this instruction snippet is easy to get from `.get_format_instructions()` of a `StructuredOutputParser`.

```python

  # How you would like your response structured. This is basically a fancy prompt template
  # Like API description for functions, this is Key descriptions of wanted dictionary.
  response_schemas = [
      ResponseSchema(name="english", description="// This a sentence in english."),
      ResponseSchema(name="german", description=" // This e sentence in german.")
  ]
  
  # How you would like to parse your output
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  
  print(output_parser.get_format_instructions())

```

Once we have the instruction snippet, lets call it `format_instructions`. We can easily form the formatted input and query the model for output.

```python

    task_instructions = "You will be given a english or german sentence from the user. please translate it and make sure they are correctly spelled in both languages."

    template = """
        {task_instructions}
        {format_instructions}
        
        % USER INPUT:
        {input}
        
        YOUR RESPONSE:
    """

    prompt = PromptTemplate(
        input_variables=["user_input"],
        partial_variables={"format_instructions": format_instructions,
                           "task_instructions": task_instructions},
        template=template
    )
    
    promptValue = prompt.format(input="Guten tag!")

```

### Format the Output - Customized Object
#### Target

We want python object like:

```python

Person(name="Jack", age=12, fav_food="potato")

```

#### Implementation

First create target object
```python

from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

```
Then extract it:

```python
from langchain.chains.openai_functions import create_structured_output_chain

llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class algorithm for extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)
chain = create_structured_output_chain(Person, llm, prompt)
chain.run("jack is 12 and love potatos.")

```

It is also able to extract a sequence of customized objects.

```python 
  from typing import Sequence
  
  class Persons(BaseModel):
      """Identifying information about all people in a text."""
  
      people: Sequence[Person] = Field(..., description="The people in the text")
  
  
  class Product(str, enum.Enum):
      CRM = "CRM"
      VIDEO_EDITING = "VIDEO_EDITING"
      HARDWARE = "HARDWARE"
      
  class Products(BaseModel):
      """Identifying products that were mentioned in a text"""
  
      products: Sequence[Product] = Field(..., description="The products mentioned in a text")
      
```

### Create Embeddings DB

#### Target

Suppose I have a very long `String` or a list of `Documents`, 
I want to shorten them by splitting them into chuncks of documents, 
and use a Vector DB to store the their embeddings. 
In order to efficiently **retrieve** similar documents.

```python
docs = retriever.get_relevant_documents("what types of things did the author want to build?")
```


#### Implementation

```python
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents([a_long_string])
# texts = text_splitter.split_documents(documents)
# P.S. texts are Documents

# Embedd your texts
db = FAISS.from_documents(texts, embeddings)

```

We can use db to retrieve similar page content chunk:

```python

retriever = db.as_retriever()
docs = retriever.get_relevant_documents("what types of things did the author want to build?")

print("\n\n".join([x.page_content[:200] for x in docs[:2]]))
```
