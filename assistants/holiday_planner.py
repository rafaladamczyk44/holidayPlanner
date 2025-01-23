from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

from models.travel_plan import TravelPlan
from config.settings import HUGGINGFACEHUB_API_TOKEN, MODEL_SELECTION, MODEL_PARAMS
from prompts.flights_prompt import get_flight_search_prompt


class HolidayPlanAssistant:
    def __init__(self):
        self.llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            repo_id=MODEL_SELECTION,
            **MODEL_PARAMS,
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.search = DuckDuckGoSearchRun()

        # Prompts
        self.flight_search_prompt = get_flight_search_prompt()

    def search_flights(self, origin:str, destination:str, date:str, n_days:int) -> str:

        search_queries = [
            f"flights from {origin} to {destination} on {date} site:kiwi.com",
            f"flights from {origin} to {destination} on {date} site:skyscanner.com"
        ]

        search_results = []

        for query in search_queries:
            results = self.search.run(query)
            search_results.append(results)

        combined_results = "\n".join(search_results)

        chain = self.flight_search_prompt | self.llm

        return chain.invoke({
            "origin": origin,
            "destination": destination,
            "date": date,
            "n_days": n_days,
            "search_results": combined_results
        })


test = HolidayPlanAssistant()
print(test.search_flights('Warsaw', 'Madrid',  '25.02.2025', 6))