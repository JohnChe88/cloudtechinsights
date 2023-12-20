import os
from langchain.llms.bedrock import Bedrock

# Initialize the Bedrock model
llm = Bedrock(model_id="anthropic.claude-v2")

prompt = "What is the largest continent?"
#prompt = (
#    "To find the largest continent, consider the size of each continent. "
#    "Asia, Africa, North America, South America, Antarctica, Europe, and Australia are the continents. "
#    "Compare their land areas to determine which is the largest."
#    "\n\nWhat is the largest continent?"
#)

response_text = llm.predict(prompt) #return a response to the prompt

print(response_text)
