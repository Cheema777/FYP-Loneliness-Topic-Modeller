import pandas as pd

# Load original dataset
df = pd.read_excel('data/Whatisloneliness060321.xlsx')

# Filter out empty or numeric rows (like -99)
df = df[df['Q9_Loneliness_Meaning_Qual'].apply(lambda x: isinstance(x, str) and len(x) > 10)]

# Take a random sample of 200 responses
test_df = df.sample(n=200, random_state=42)[['Idme', 'Q9_Loneliness_Meaning_Qual']].copy()
test_df.columns = ['ID', 'Text']

# Leave True_Theme blank for manual labeling
test_df['True_Theme'] = ""

# Save to CSV
test_df.to_csv('data/test_dataset.csv', index=False)
print("Saved 200 blank rows to data/test_dataset.csv for manual labeling!")
