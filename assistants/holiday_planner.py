from typing import Dict
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

from models.travel_plan import TravelPlan
from config.settings import HUGGINGFACEHUB_API_TOKEN, MODEL_SELECTION, MODEL_PARAMS


class HolidayPlanAssistant:
    def __init__(self, hf_token_id: str):
        self.llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            repo_id=MODEL_SELECTION,
            **MODEL_PARAMS,
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.search = DuckDuckGoSearchRun()

    def search_flights(self, origin:str, destination:str, date:str, n_days:int) -> Dict:
        flight_prompt = PromptTemplate(
            input_variables = ['origin', 'destination', 'date', 'n_days', 'search_results'],
            template = """
                Based on the search results, analyze flights option from {origin} to {destination}, on {date}, with a return flight in {n_days} days.
                Only return direct flights, 
                You are allowed to slightly modify the dates if there are no flights matching in the selected dates.
                Limit the results to the two best options.
                
                Please use a scoring system for the flights: 
                    1. the closer the flight is to selected date, the better
                    2. flights with the smallest amount of landings are preferred (direct flights are the best)
                    3. start with cheaper flights
                    
                Provide a structured analysis of the best flight options in the following format:
                    1. Airline, Price
                    2. Departure date and time from {origin}, flight time, airport
                    3. Departure date and time from {destination}, flight time, airport
                    4. Baggage policy
                    
                Return the information in a clear, organized manner. Please use PLN as a currency
                    Search Results: 
                    {search_results}
            """
        )
        search_queries = [
            f"flights from {origin} to {destination} on {date} site:kiwi.com",
            f"flights from {origin} to {destination} on {date} site:skyscanner.com"
        ]

        search_results = []
        for query in search_queries:
            results = self.search.run(query)
            search_results.append(results)

        combined_results = "\n".join(search_results)

        # Create and use the chain properly
        chain = LLMChain(llm=self.llm, prompt=flight_prompt)

        # Use run() method with keyword arguments
        return chain.run(
            origin=origin,
            destination=destination,
            date=date,
            n_days=n_days,
            search_results=combined_results
        )


test = HolidayPlanAssistant(HUGGINGFACEHUB_API_TOKEN)
print(test.search_flights('Warsaw', 'Madrid',  '25.02.2025', 6))