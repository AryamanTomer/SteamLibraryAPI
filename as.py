import pandas as pd
df = pd.read_csv("steam_library_for_model.csv")
pd.DataFrame({
    "Name": df["Name"],
    "Hours_Main_Story": "",
    "Genre1": "",
    "Genre2": "",
    "Year Released": "",
    "Developer": "",
    "Publisher": "",
}).to_csv("hltb_placeholder.csv", index=False)
