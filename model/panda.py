import pandas as pd
df = pd.read_csv("data/isl_landmark_data.csv")
print(df['label'].value_counts())
