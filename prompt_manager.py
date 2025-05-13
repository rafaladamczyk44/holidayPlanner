import yaml
from pathlib import Path
from langchain_core.prompts import PromptTemplate

class PromptManager:
    def __init__(self,  prompts_directory='prompts'):
        self.prompts_directory = Path(prompts_directory)

    def return_prompt(self, name:str) -> PromptTemplate:
        path = self.prompts_directory / f'{name}.yaml'

        with open(path, 'r') as f:
            prompt_data = yaml.safe_load(f)

        template = prompt_data.get('template', '')
        variables = prompt_data.get('variables', [])

        return PromptTemplate(template=template, input_variables=variables)
