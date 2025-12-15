import pandas as pd

X_test = pd.read_csv(
    "/Users/anasnasr/Library/CloudStorage/OneDrive-FutureUniversityinEgypt/UCI HAR Dataset/test/X_test.txt",
    sep=r"\s+",
    header=None
)

# Take only ONE sample
X_test_one = X_test.iloc[[0]]

X_test_one.to_csv("X_test_one_row.csv", index=False)
print("X_test_one_row.csv created")
