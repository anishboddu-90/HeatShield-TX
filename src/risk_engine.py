def calculate_risk_score(simulated_costs, user_budget):
    sim = list(simulated_costs)
    if len(sim) == 0:
        return 0.0
    counter = sum(1 for v in sim if v > user_budget)
    risk_score = counter / len(sim) * 100
    return risk_score

def calculate_risk_score(simulated_costs, user_budget):
    counter = 0
    for i in range(len(simulated_costs)):
        if simulated_costs[i] > user_budget:
            counter +=1
    risk_score = counter / len(simulated_costs) * 100
    return risk_score