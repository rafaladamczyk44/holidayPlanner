import os
import dotenv
import logging
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_google_community import GoogleSearchAPIWrapper

from prompt_manager import PromptManager

dotenv.load_dotenv('config.env')

MODEL_SELECTION = os.getenv('MODEL_SELECTION')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# TODO: Search to be rewritten using Amadeus

class HolidayPlanAssistant:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID,
            k=5
        )

        # Using phi-3-mini
        self.llm = OllamaLLM(
            model=MODEL_SELECTION,
            temperature=0.6,
            top_p=0.9,
            repeat_penalty=1.1,
            num_predict=512,
        )

        # Prompts
        self.prompt_manager = PromptManager()


    def search_queries(self, search_q) -> Optional[str]:
        """
        Method for organizing browser searching
        Args:
            search_q str: (List) of string for query search
        Returns:
            Search results
        """
        results = []

        for query in search_q:

            search_result = self.search.run(query)

            if search_result:  # Check if results are not empty
                results.append(search_result)
            else:
                print(f"No results found for query: {query}")
                return None


        return "\n".join(results)


    def search_flights(self, origin:str, destination:str, date:str, n_days:int) -> str:
        """
        Args:
            origin (str): Origin city
            destination (str): Destination city
            date (str): Date of flights
            n_days (int): How long to stay
        Returns:
            LLM Generated flights overview
        """

        # Create more specific search queries
        search_queries = [
            f"site:kiwi.com flights {origin} to {destination}",
            f"site:skyscanner.com flights {origin} to {destination}",
        ]

        combined_results = self.search_queries(search_queries)

        prompt = self.prompt_manager.return_prompt('flight')

        chain = prompt | self.llm

        agent_response = chain.invoke({
            "origin": origin,
            "destination": destination,
            "date": date,
            "n_days": n_days,
            "search_results": combined_results
        })

        return agent_response

