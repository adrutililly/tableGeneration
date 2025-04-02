
from azure.identity import ClientSecretCredential, get_bearer_token_provider
import os
from dotenv import load_dotenv




# os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("aws_access_key_id")
# os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("aws_secret_access_key")
# os.environ["AWS_SESSION_TOKEN"] = os.getenv("aws_session_token")


# load_dotenv()




tenant_id = "18a59a81-eea8-4c30-948a-d8824cdc2580"
client_secret = ""
client_id = "9ca21d06-5810-4e98-8cb1-045d056d354d"





token_provider = get_bearer_token_provider(
            ClientSecretCredential(tenant_id, client_id, client_secret),
            "https://cognitiveservices.azure.com/.default"
            )
from langchain_openai import AzureChatOpenAI

# if model_type == "aws":

# else:
llm = AzureChatOpenAI(
    model="gpt-4omni-latest", # gpt-4omni-latest, gpt-4o-mini-2024-07-18
    # model = "gpt-4o",
    api_version = "2024-10-21", # 2024-10-21, 2024-07-01-preview, 2024-12-01-preview
    azure_endpoint = "https://gis-policy-instance-dev.openai.azure.com/", # "https://gis-policy-instance-eus-dev.openai.azure.com/", #openai/deployments/gpt-4o-mini-2024-07-18/chat/completions?api-version=2023-03-15-preview",
    azure_ad_token_provider = token_provider,
    temperature=0 
    )

