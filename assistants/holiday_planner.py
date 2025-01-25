from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

# from models.travel_plan import TravelPlan
from config.settings import HUGGINGFACEHUB_API_TOKEN, MODEL_SELECTION, MODEL_PARAMS
from prompts.flights_prompt import get_flight_search_prompt
from prompts.accomodation_prompt import get_accomodation_search_prompt


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
        self.accommodation_prompt = get_accomodation_search_prompt()

    def search_queries(self, search_q) -> str:
        """
        Method for organizing browser searching
        :param search_q: Query to be searched
        :return: String
        """

        search_results = []
        try:
            for query in search_q:
                results = self.search.run(query)
                search_results.append(results)
        except DuckDuckGoSearchException:
            print("Can't search at the moment :(")

        return "\n".join(search_results)

    def search_flights(self, origin:str, destination:str, date:str, n_days:int) -> str:
        """
        Ask agent to return list of the best fitting flights in given dates based on a scoring system
        (Price, how close it is to the selected dates, baggage policy).
        :param origin: City to depart from
        :param destination: City of final destination
        :param date: Flight date
        :param n_days: Days before return
        :return: String
        """
        search_queries = [
            f"flights from {origin} to {destination} on {date} site:kiwi.com",
            f"flights from {origin} to {destination} on {date} site:skyscanner.com"
        ]

        combined_results = self.search_queries(search_queries)

        chain = self.flight_search_prompt | self.llm

        agent_response = chain.invoke(
            {
                "origin": origin,
                "destination": destination,
                "date": date,
                "n_days": n_days,
                "search_results": combined_results
            }
        )

        return agent_response

    def search_accommodations(self, destination:str, date:str, n_days:int) -> str:
        search_queries = [
            f'Accommodation in {destination} on {date}, for {n_days} nights site:booking.com',
        ]

        combined_results = self.search_queries(search_queries)

        chain = self.accommodation_prompt | self.llm
        agent_response = chain.invoke(
            {
                "destination": destination,
                "date": date,
                "n_days": n_days,
                "search_results": combined_results
            }
        )

        return agent_response

    def concat_search_results(self, search_results_flight, search_results_accommodations):
        pass
