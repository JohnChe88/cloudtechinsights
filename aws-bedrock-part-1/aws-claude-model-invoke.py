import os
import json
import boto3

# Initialize a session with AWS. This sets up the credentials for AWS access.
# The credentials are typically sourced from environment variables or configuration files.
session = boto3.Session()

# Create a client for the 'bedrock-runtime' service. This service is used to interact with AWS Bedrock.
bedrock = session.client(service_name='bedrock-runtime') 

# Define the parameters for invoking the model.
# 'prompt_text' is the input text for the model, which should be formatted as a dialog.
prompt_text = "Human: Which is the largest continent\nAssistant:"

prompt_params = {
    "prompt": prompt_text,# 'prompt' is the input text for the model.
    "max_tokens_to_sample": 4096, # 'max_tokens_to_sample' defines the maximum length of the model's response.
    "temperature": 0.5,# 'temperature' controls the randomness of the response. Lower values make responses more predictable.
    "top_k": 250,# 'top_k' limits the model's choices to the top K most likely next words.
    "top_p": 0.5,  # 'top_p' is an alternative to top_k, where the model considers a dynamic number of words.
    "stop_sequences": ["\n"] # 'stop_sequences' are tokens at which the model will stop generating further content.
}

# Convert the prompt parameters to a JSON string and then encode it to bytes.
# This is required as the Bedrock API expects the payload in this format.
body = json.dumps(prompt_params).encode('utf-8')

# Invoke the model using the Bedrock client. 
# The 'modelId' parameter specifies the model to use, in this case, 'anthropic.claude-v2'.
# 'accept' and 'contentType' specify the format of the request and response.
response = bedrock.invoke_model(
    body=body,
    modelId="anthropic.claude-v2", 
    accept='application/json', 
    contentType='application/json'
)

# Process the response from the model.
# The response body is parsed from JSON, and the completion text (model's response) is extracted.
response_body = json.loads(response.get('body').read()) 
response_text = response_body.get("completion")

# Print the response text.
print(response_text)
