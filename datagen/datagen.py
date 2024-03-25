import os
import time
import pandas as pd
from functools import partial
import config
from datasets import Dataset
from client.r2d2_client import R2D2Client
from datagen.utils import read_files
from datagen.prompt import find_prompt
from datagen.dataprep import convert_to_text, preprocess
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

def question_gen(docs, bare_template, gen_provider):
    print("Generating questions")
    print(f"Using {gen_provider}")

    question_response_schemas = [
        ResponseSchema(name="question", description="question about the context."),
    ]
    
    question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
    format_instructions = question_output_parser.get_format_instructions()

    question_generation_llm = R2D2Client().get_gen_client(gen_provider=gen_provider)

    qa_template = find_prompt("system", "professor_q")

    print(f"Using template: {qa_template}")

    prompt_template = ChatPromptTemplate.from_template(template=qa_template)

    messages = prompt_template.format_messages(
        context=docs[0],
        format_instructions=format_instructions
    )

    question_generation_chain = bare_template | question_generation_llm

    qac_triples = []

    for text in docs:
        messages = prompt_template.format_messages(
            context=text,
            format_instructions=format_instructions
        )
        try:
            if gen_provider == "azure":
                response = question_generation_chain.invoke({"content": messages})
                output_dict = question_output_parser.parse(response.content)
            elif gen_provider == "vertex":
                response = question_generation_chain.invoke({"content": messages})
                output_dict = question_output_parser.parse(response.content)
        except Exception as e:
            print(e)
            output_dict = {
                "question": "",
                "text": text,
            }
        qac_triples.append(output_dict)
    
    return qac_triples

def answer_gen(qac_triples, bare_template, gen_provider):
    print("Generating answers")
    print(f"Using {gen_provider}")

    answer_generation_llm = R2D2Client().get_gen_client(gen_provider=gen_provider)

    answer_schema = ResponseSchema(
        name="answer",
        description="An answer to the question"
    )

    answer_response_schemas = [
        answer_schema,
    ]

    answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
    format_instructions = answer_output_parser.get_format_instructions()

    qa_template = find_prompt("system", "professor_a")

    print(f"Using template: {qa_template}")

    prompt_template = ChatPromptTemplate.from_template(template=qa_template)

    messages = prompt_template.format_messages(
        context=qac_triples[0]["context"],
        question=qac_triples[0]["question"],
        format_instructions=format_instructions
    )

    answer_generation_chain = bare_template | answer_generation_llm

    for triple in qac_triples:
        messages = prompt_template.format_messages(
            context=triple["context"],
            question=triple["question"],
            format_instructions=format_instructions
        )
        try:
            response = answer_generation_chain.invoke({"content": messages})
            output_dict = answer_output_parser.parse(response.content)
            print(output_dict)
        except Exception as e:
            continue
        triple["ground_truth"] = output_dict["answer"]

    return qac_triples

def ground_truth_gen(qac_triples):
    print("Setting ground truth")
    try:
        ground_truth_qac_set = pd.DataFrame(qac_triples)
        ground_truth_qac_set = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
    except Exception as e:
        print(e)
        ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer": "ground_truth"})

    return ground_truth_qac_set

def get_metadata(file_path):
    return {"file_path": file_path}
