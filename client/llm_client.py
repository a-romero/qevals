import os
from pyexpat import model
import subprocess
from sys import api_version

from networkx import project
from Delphic.config.api.auth import api_key
import config

import vertexai
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_google_vertexai import VertexAI
from google.oauth2.credentials import Credentials
from openai import AzureOpenAI

os.environ['SSL_CERT_FILE'] = config.config['MISC']['SSL_CERT_FILE']

class LLMClient:
    def __init__(self):
        self.api_key = config.config['MISC']['API_KEY']

    def get_openai_client(self, function) -> AzureChatOpenAI:
        client = AzureChatOpenAI(
            api_key=self.api_key,
            azure_endpoint=config.config[function]['AZURE_ENDPOINT'],
            model=config.config[function]['MODEL'],
            api_version=config.config[function]['API_VERSION']
            )
        return client
    
    def get_vertexai_client(self, function) -> VertexAI:
        vertexai.init(
            api_key=self.api_key,
            project=config.config[function]['PROJECT'],
            api_transport='rest',
            api_endpoint=config.config[function]['VERTEX_ENDPOINT'],
        )
        client = VertexAI(
            model_name=config.config[function]['MODEL'],
            )
        return client