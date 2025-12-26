class PlannerAgent:
    def __init__(self):
        # We could initialize an LLM here if we wanted complex planning,
        # but for this assignment, we use simple keyword logic as requested.
        pass

    def plan(self, state: dict) -> dict:
        """
        Analyzes the user query and determines the best approach.
        Updates 'plan' and 'query_type' in the state.
        """
        query = state.get("query", "").lower()
        print(f"\n--- [Planner] Analyzing query: '{state.get('query')}' ---")

        # Simple keyword logic for query classification
        if any(w in query for w in ["compare", "difference", "versus", "vs"]):
            query_type = "comparison"
            plan = "Search for both concepts and highlight distinctions."
        elif any(w in query for w in ["how to", "steps", "guide", "process"]):
            query_type = "how-to"
            plan = "Look for procedural steps or implementation details."
        elif any(w in query for w in ["what is", "define", "meaning", "concept"]):
            query_type = "definition"
            plan = "Find the exact definition and core concepts."
        elif any(w in query for w in ["best practice", "recommendation", "tip"]):
            query_type = "recommendation"
            plan = "Search for architectural patterns and AWS recommendations."
        else:
            query_type = "general"
            plan = "Retrieve general context relevant to the query."

        print(f"Decision: Query Type = {query_type}")
        print(f"Plan: {plan}")
        
        # Return only the updates to the state
        return {
            "query_type": query_type,
            "plan_description": plan
        }
