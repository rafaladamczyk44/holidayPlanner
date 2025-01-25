from langchain_core.prompts import PromptTemplate

ACCOMMODATION_SEARCH_TEMPLATE = """
Your task it to find the best accommodation in the {destination}, on {date}, for {n_days} nights.
The best place is the one that passes all the checks:
1. It is clode to the city centre or to the tourist area, so that all attractions or activites are a short walk/commute from it
2. You should focus on apartments first, then hotels and then rooms or hostels
3. You always look at the cheapest first, if it follows all the rules, this is the one.

Provide a structured analysis of the best accommodations in the following format:
1. Price per person per night
2. Area
3. Type of objcet
4. Check-in, check-out hours
5. Summarized description of the place (50 words max)

Return the information in a clear, organized manner. Please use PLN as a currency
Search Results:
{search_results}
"""

def get_accomodation_search_prompt() -> PromptTemplate:
    prompt = PromptTemplate(
        input_variables=[
            'destination',
            'date',
            'n_days',
            'search_results',
        ],
        template=ACCOMMODATION_SEARCH_TEMPLATE,
    )
    return prompt