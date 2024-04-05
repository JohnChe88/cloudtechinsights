import boto3
import json
import time

def invoke_bedrock_model(english_input):
    """
    Invokes the Bedrock model to convert English input to SQL query.
    
    Parameters:
    - english_input: The English sentence to be converted into SQL.
    
    Returns:
    - SQL query as a string.
    """
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime') 
    #pre_context="users information is stored in table name is s3regulartable, it has columns name for username,age and city is where user lives and it is in database cloudtechinsights"
    
    pre_context="All the tables information is provided in JSON format"
    # Define the metadata as a Python list of dictionaries
    json_metadata = [
    {
        "table_name": "s3regulartable",
        "database":"cloudtechinsights",
        "description": "has data about all the users",
        "columns": [
            {"name": "Name", "type": "string", "description": "Name of the user"},
            {"name": "Age", "type": "integer", "description": "Age of the user"},
            {"name": "City", "type": "string", "description": "City where the user lives"},
        ],
        "update_frequency": "daily",
        "usage": "Used for analyzing user details."
    },
    {
        "table_name": "sales",
        "database":"cloudtechinsights",
        "description": "sales data for the segment",
        "columns": [
            {"name": "Name", "type": "string", "description": "Name of the segment"},
            {"name": "sales", "type": "integer", "description": "sales number"},
            {"name": "City", "type": "string", "description": "City where the segment"},
        ],
        "update_frequency": "daily",
        "usage": "Used for analyzing sales details."
    }
    ]

    # Convert the metadata to a JSON string
    json_string = json.dumps(json_metadata, indent=4)  # `indent=4` for pretty printing

    post_context="In the response prove database.table_name and  provide only SQL no other text"
    prompt_text = (" Create a SQL to get users with age greater than 10 "
                   )
    
    prompt="Human:"+pre_context+" "+json_string+" "+prompt_text+" . "+post_context+"\nAssistant:"
    
    print("\n\nprompt :  ", prompt)
    
    prompt_params = {
        "prompt": prompt,
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": ["\n\nHuman"]
    }
    body = json.dumps(prompt_params).encode('utf-8')
    
    response = bedrock.invoke_model(
        body=body,
        modelId="anthropic.claude-v2", 
        accept='application/json', 
        contentType='application/json'
    )
    response_body = json.loads(response.get('body').read()) 
    response_text = response_body.get("completion")

    #response_body = json.loads(response['payload'].read().decode("utf-8")) 
    return response_body.get("completion")

def execute_athena_query(sql_query):
    """
    Executes the given SQL query in Amazon Athena and waits for the result.
    
    Parameters:
    - sql_query: The SQL query to be executed.
    
    Returns:
    - The result of the query execution.
    """
    athena_client = boto3.client('athena')
    
    query_config = {
        'QueryString': sql_query,
        'QueryExecutionContext': {'Database': 'cloudtechinsights'},
        'ResultConfiguration': {'OutputLocation': 's3://audio-files-ytube/'}
    }
    
    response = athena_client.start_query_execution(**query_config)
    query_execution_id = response['QueryExecutionId']
    return wait_for_athena_query_result(athena_client, query_execution_id)

def wait_for_athena_query_result(athena_client, query_execution_id):
    """
    Waits for an Athena query to complete and fetches the result.
    
    Parameters:
    - athena_client: The Athena client object.
    - query_execution_id: The ID of the query execution to wait for.
    
    Returns:
    - The result of the query execution.
    """
    while True:
        query_status = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = query_status['QueryExecution']['Status']['State']
        
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        else:
            time.sleep(5)
    
    if state == 'SUCCEEDED':
        return athena_client.get_query_results(QueryExecutionId=query_execution_id)
    else:
        raise Exception(f'Query execution failed with state {state}')

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Parameters:
    - event: The event triggering this Lambda function.
    - context: The execution context of this Lambda function.
    
    Returns:
    - The response object containing the status code and body.
    """
    user_input = event.get('user_input_english')
    sql_query = invoke_bedrock_model(user_input)
    
    print("sql-query : ",sql_query)
    
    if not sql_query:
        return {'statusCode': 400, 'body': json.dumps({'message': 'Failed to translate English to SQL'})}
    
    try:
        query_result = execute_athena_query(sql_query)
        return {'statusCode': 200, 'body': json.dumps({'message': 'Query executed successfully', 'data': str(query_result)})}

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'message': str(e)})}
