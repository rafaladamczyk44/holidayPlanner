from langchain_core.prompts import PromptTemplate

FLIGHT_SEARCH_TEMPLATE = """
Based on the search results, analyze flights option from {origin} to {destination}, on {date}, with a return flight in {n_days} days.
Only direct flights, 
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


def get_flight_search_prompt() -> PromptTemplate:
    prompt = PromptTemplate(
        input_variables=[
            'origin',
            'destination',
            'date',
            'n_days',
            'search_results'
        ],
        template=FLIGHT_SEARCH_TEMPLATE
    )
    return prompt