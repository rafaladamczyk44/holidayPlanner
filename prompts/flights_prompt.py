from langchain_core.prompts import PromptTemplate

FLIGHT_SEARCH_TEMPLATE = """
    Based on the search results, analyze flight options from {origin} to {destination}, on {date}, with a return flight in {n_days} days.
    
    Use the following scoring system (0-10 points for each category):

    1. Schedule Score (max 10 points):
        - Exact date match: 10 points
        - +/- 1 day: 7 points
        - +/- 2 days: 4 points
        - Departure time between 10:00-18:00: 10 points
        - Departure time 8:00-10:00 or 18:00-21:00: 7 points
        - Departure time before 8:00 or after 21:00: 3 points

    2. Value Score (max 10 points):
        - Base price comparison (relative to average price):
            * Below average: 8-10 points
            * Average: 5-7 points
            * Above average: 1-4 points
        - Baggage policy:
            * Included checked baggage: +2 points
            * Only cabin baggage: +0 points
            * Extra fees for cabin baggage: -2 points

    3. Flight Quality Score (max 10 points):
        - Direct flight: 10 points
        - 1 stop: 5 points
        - 2+ stops: 0 points
        - Duration penalty: -1 point for each hour above average flight time
        - Major airline: +2 points
        - Budget airline: +0 points

    Calculate the final score as weighted average:
        - Schedule Score: 40%
        - Value Score: 35%
        - Flight Quality Score: 25%

    Return the top 5 flights with highest total scores. For each flight provide:
        1. Total Score (out of 10)
        2. Airline, Price (in PLN)
        3. Departure date and time, flight duration, airports (first flight)
        4. Departure date and time, flight duration, airports (return flight)
        4. Baggage policy

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