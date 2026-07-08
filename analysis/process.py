import pandas as pd
import numpy as np
import json
import os

df = pd.read_csv("titanic.csv")
print("=== Original Shape ===", df.shape)

# ---------- CLEANING ---------- #
# Drop PassengerId (index), Ticket (too many unique), Name (too many unique)
df_clean = df.drop(columns=["PassengerId", "Ticket", "Name"])

# Create Cabin flag
df_clean["HasCabin"] = df_clean["Cabin"].notna().astype(int)
df_clean = df_clean.drop(columns=["Cabin"])

# Fill Embarked with mode
embarked_mode = df_clean["Embarked"].mode()[0]
df_clean["Embarked"] = df_clean["Embarked"].fillna(embarked_mode)

# Fill Age with median by Sex and Pclass
df_clean["Age"] = df_clean.groupby(["Sex", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Any remaining missing
df_clean = df_clean.dropna()

print("=== Cleaned Shape ===", df_clean.shape)
print("=== Missing after clean ===")
print(df_clean.isnull().sum())

# Save cleaned data
df_clean.to_csv("analysis/titanic_clean.csv", index=False)

# ---------- ANALYSIS ---------- #
# 1. Survival rate by gender
surv_by_sex = df_clean.groupby("Sex")["Survived"].agg(["count", "mean"]).round(4)
surv_by_sex_dict = surv_by_sex.reset_index().to_dict(orient="records")

# 2. Survival rate by class
surv_by_class = df_clean.groupby("Pclass")["Survived"].agg(["count", "mean"]).round(4)
surv_by_class_dict = surv_by_class.reset_index().to_dict(orient="records")

# 3. Age distribution
age_bins = [0, 10, 20, 30, 40, 50, 60, 100]
age_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
df_clean["AgeGroup"] = pd.cut(df_clean["Age"], bins=age_bins, labels=age_labels, right=False)
age_survival = df_clean.groupby("AgeGroup", observed=True)["Survived"].agg(["count", "mean"]).round(4)
age_survival_dict = age_survival.reset_index().to_dict(orient="records")

# 4. Fare distribution stats
fare_stats = {
    "min": round(float(df_clean["Fare"].min()), 2),
    "max": round(float(df_clean["Fare"].max()), 2),
    "mean": round(float(df_clean["Fare"].mean()), 2),
    "median": round(float(df_clean["Fare"].median()), 2),
    "std": round(float(df_clean["Fare"].std()), 2),
}

# 5. SibSp / Parch -> FamilySize
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
family_survival = df_clean.groupby("FamilySize")["Survived"].agg(["count", "mean"]).round(4)
family_survival_dict = family_survival.reset_index().to_dict(orient="records")

# 6. Survival by Embarked
surv_by_embarked = df_clean.groupby("Embarked")["Survived"].agg(["count", "mean"]).round(4)
surv_by_embarked_dict = surv_by_embarked.reset_index().to_dict(orient="records")

# 7. Survival by HasCabin
surv_by_cabin = df_clean.groupby("HasCabin")["Survived"].agg(["count", "mean"]).round(4)
surv_by_cabin_dict = surv_by_cabin.reset_index().to_dict(orient="records")

# 8. Correlation matrix (numeric only)
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin", "FamilySize"]
corr_matrix = df_clean[numeric_cols].corr().round(4)
corr_dict = {
    "labels": corr_matrix.columns.tolist(),
    "values": corr_matrix.values.tolist(),
}

# 9. Overall stats
overall = {
    "total": int(len(df_clean)),
    "survived": int(df_clean["Survived"].sum()),
    "died": int((df_clean["Survived"] == 0).sum()),
    "survival_rate": round(float(df_clean["Survived"].mean()), 4),
}

# Gather results
results = {
    "overall": overall,
    "survival_by_sex": surv_by_sex_dict,
    "survival_by_class": surv_by_class_dict,
    "survival_by_age_group": age_survival_dict,
    "fare_stats": fare_stats,
    "survival_by_family_size": family_survival_dict,
    "survival_by_embarked": surv_by_embarked_dict,
    "survival_by_cabin": surv_by_cabin_dict,
    "correlation": corr_dict,
}

os.makedirs("analysis", exist_ok=True)
with open("analysis/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n=== Analysis Results saved to analysis/results.json ===")
print(json.dumps(overall, indent=2))
