# AWS Bedrock Agent Setup and Invocation

This repository contains Python code to create and interact with an AWS Bedrock Agent. The Bedrock Agent is set up to handle tasks such as code execution, code generation, and data analysis. The provided script includes the setup of IAM roles and policies, agent creation, configuration of a code interpreter, and invoking the agent.

## Prerequisites

Before running the script, ensure you have the following:

- AWS account with necessary permissions to create IAM roles and policies.
- AWS CLI configured with appropriate credentials.
- Python 3.x installed.
- `boto3` library installed. You can install it using:

```bash
pip install boto3
