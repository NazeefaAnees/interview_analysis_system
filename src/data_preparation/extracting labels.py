import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset that includes all your features and the target overall score
# (Replace 'master_dataset_structured_features.csv' with your file name if needed)
df = pd.read_csv('../../data/test-final/master_dataset_structured_features.csv')

# Let's assume your target column is named 'Overall_Score' (or you can use 'overall_scores.overall_suitability' if that's the intended target)
# For this example, I'll assume it's called 'Overall_Score'. If it's nested, you may need to extract it.
# If your file contains 'Overall_Suitability' as a feature from the questionnaire and another overall score column, 
# use the one representing the final candidate evaluation (overall_sscore).
# For example, we can assume it is 'Overall_Score'.

# If your target column is 'overall_scores.overall_suitability', rename it for clarity:
if 'overall_scores.overall_suitability' in df.columns:
    df.rename(columns={'overall_scores.overall_suitability': 'Overall_Score'}, inplace=True)

# Ensure the target is numeric
df['Overall_Score'] = df['Overall_Score'].astype(float)

# Visualize the distribution of the target overall score
plt.hist(df['Overall_Score'], bins=10, edgecolor='black')
plt.title("Distribution of Overall Score (Target)")
plt.xlabel("Overall Score")
plt.ylabel("Frequency")
plt.show()

print("Summary statistics for Overall_Score:")
print(df['Overall_Score'].describe())

# Save the dataset with the properly labeled target for later training
df.to_csv('../../data/test-final/master_dataset_for_training.csv', index=False)
print("Dataset for training (with target Overall_Score) saved as 'master_dataset_for_training.csv'")
