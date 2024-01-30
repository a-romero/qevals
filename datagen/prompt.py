import csv

PROMPTS_TEMPLATE = './datagen/prompts/template.csv'

def find_prompt(target_role, target_function):
    csv_file = PROMPTS_TEMPLATE
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            role, function, prompt = row
            if role == target_role and function == target_function:
                return prompt
    return None
