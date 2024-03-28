import os
from functools import partial

from client.llm_client import LLMClient
from datagen.utils import files
from datagen.prompt import find_prompt
from datagen.dataprep import convert_to_text, preprocess

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

def generation(gen_provider: str, input: str, context: str) -> str:
    question_schema = ResponseSchema(
        name="answer",
        description="an answer based on the context."
    )
    
    question_response_schemas = [
        question_schema,
    ]

    question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
    format_instructions = question_output_parser.get_format_instructions()

    gen_llm = LLMClient().get_gen_client(gen_provider)

    input_template = find_prompt("user", "gen_prompt1")
    prompt_template = ChatPromptTemplate.from_template(template=input_template)
    message = prompt_template.format_messages(
        question=input,
        context=context,
        format_instructions=format_instructions
    )

    bare_prompt_template = "{content}"
    bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)
    answer_generation_chain = bare_template | gen_llm
    response = answer_generation_chain.invoke({"content": message})

    return response
