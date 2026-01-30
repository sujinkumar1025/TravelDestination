import pandas as pd
import numpy as np
import random

# -----------------------------
# CONFIGURATION
# -----------------------------
random.seed(42)
np.random.seed(42)

# -----------------------------
# DESTINATIONS
# -----------------------------
destinations_distribution = {
    "Goa": 1527,
    "Kerala": 1345,
    "Manali": 1250,
    "Shimla": 1180,
    "Jaipur": 1400,
    "Agra": 1100,
    "Udaipur": 1050,
    "Ooty": 980,
    "Rishikesh": 1150,
    "Darjeeling": 1020,
    "Leh": 998
}

# -----------------------------
# DESTINATION CONSTRAINTS
# -----------------------------
patterns = {
    "Goa":        ("Beach","Winter","Low",(28,32),(8000,25000)),
    "Kerala":     ("Beach","Winter","Medium",(28,32),(25001,35000)),
    "Manali":     ("Adventure","Summer","Medium",(10,20),(20000,30000)),
    "Shimla":     ("Adventure","Summer","High",(15,25),(30001,40000)),
    "Jaipur":     ("Historical","Winter","High",(22,28),(35000,45000)),
    "Agra":       ("Historical","Winter","Medium",(22,28),(25000,35000)),
    "Udaipur":    ("Historical","Winter","High",(25,30),(35000,45000)),
    "Ooty":       ("Adventure","Summer","Medium",(15,20),(20000,30000)),
    "Rishikesh":  ("Adventure","Summer","Low",(18,25),(15000,25000)),
    "Darjeeling": ("Adventure","Summer","Medium",(12,20),(18000,28000)),
    "Leh":        ("Adventure","Summer","High",(5,15),(25000,40000))
}

travel_method_options = {
    "Goa": ["Flight","Train"],
    "Kerala": ["Flight","Train"],
    "Manali": ["Bus","Car"],
    "Shimla": ["Bus","Car"],
    "Jaipur": ["Train","Car"],
    "Agra": ["Train","Car"],
    "Udaipur": ["Train","Car"],
    "Ooty": ["Bus","Car"],
    "Rishikesh": ["Bus","Car"],
    "Darjeeling": ["Bus","Train"],
    "Leh": ["Flight","Bus"]
}

accommodation_options = {
    "Goa": ["Resort","Hotel"],
    "Kerala": ["Resort","Hotel"],
    "Manali": ["Lodge","Homestay"],
    "Shimla": ["Lodge","Homestay"],
    "Jaipur": ["Hotel","Resort"],
    "Agra": ["Hotel","Resort"],
    "Udaipur": ["Hotel","Resort"],
    "Ooty": ["Lodge","Homestay"],
    "Rishikesh": ["Lodge","Homestay"],
    "Darjeeling": ["Lodge","Homestay"],
    "Leh": ["Lodge","Homestay"]
}

climate_options = {
    "Goa": ["Warm"],
    "Kerala": ["Warm"],
    "Manali": ["Cold"],
    "Shimla": ["Cold"],
    "Jaipur": ["Hot"],
    "Agra": ["Hot"],
    "Udaipur": ["Hot"],
    "Ooty": ["Cold"],
    "Rishikesh": ["Warm"],
    "Darjeeling": ["Cold"],
    "Leh": ["Cold"]
}

min_duration_dict = {
    "Goa": 1, "Kerala": 2, "Manali": 2, "Shimla": 2, "Jaipur": 1,
    "Agra": 1, "Udaipur": 1, "Ooty": 2, "Rishikesh": 2,
    "Darjeeling": 3, "Leh": 5
}

crowd = ["Low","Medium","High"]
family_friendly = ["Yes","No"]

# -----------------------------
# DATA GENERATION
# -----------------------------
data = []

for dest, count in destinations_distribution.items():
    t_type, season, budget, temp_range, cost_range = patterns[dest]
    min_duration = min_duration_dict[dest]

    for _ in range(count):
        # Deterministic temp within range (min to max only)
        temp = random.randint(temp_range[0], temp_range[1])
        cost = random.randint(*cost_range)

        duration = random.randint(min_duration, min_duration + 8)
        rating = round(random.uniform(1.0, 5.0), 2)

        travel_method = random.choice(travel_method_options[dest])
        accommodation = random.choice(accommodation_options[dest])
        climate = random.choice(climate_options[dest])

        data.append([
            dest, budget, climate, t_type, season, duration,
            f"{temp}°C", f"₹{cost}", rating,
            random.choice(crowd), accommodation,
            random.choice(family_friendly), travel_method
        ])

# -----------------------------
# ADD OUTLIERS (same as before)
# -----------------------------
for _ in range(180):
    data[random.randint(0, len(data)-1)][7] = f"₹{random.randint(150000,300000)}"

for _ in range(90):
    data[random.randint(0, len(data)-1)][6] = f"{random.randint(-15,60)}°C"

for _ in range(60):
    data[random.randint(0, len(data)-1)][5] = random.randint(20,45)

for _ in range(40):
    data[random.randint(0, len(data)-1)][8] = round(random.uniform(1.0,5.0),2)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
columns = [
    "Destination","Budget","Climate","Travel_Type","Best_Season",
    "Duration_Days","Avg_Temperature","Avg_Cost","Rating",
    "Crowd_Level","Accommodation_Type","Family_Friendly","Travel_Method"
]

df = pd.DataFrame(data, columns=columns)

# -----------------------------
# ADD MISSING VALUES (same as before)
# -----------------------------
missing_config = {
    "Avg_Cost": 125,
    "Avg_Temperature": 73,
    "Budget": 98,
    "Family_Friendly": 60,
    "Travel_Method": 42
}

for col, count in missing_config.items():
    idxs = df.sample(n=count, random_state=random.randint(1,999)).index
    df.loc[idxs, col] = np.nan

# -----------------------------
# SAVE DATASET
# -----------------------------
df.to_csv("travel_syntetic6_realistic_strict.csv", index=False)
print("✅ Fully deterministic realistic dataset created:", df.shape)
