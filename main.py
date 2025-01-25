from assistants.holiday_planner import HolidayPlanAssistant

origin = 'Warsaw'
destination = 'Madrid'
date = '11/04/2025'
n_days = 7

if __name__ == '__main__':
    assistant = HolidayPlanAssistant()
    flights = assistant.search_flights(origin, destination, date, n_days)
    accommodation = assistant.search_accommodations(destination, date, n_days)
     # TODO: Combine results into one reply