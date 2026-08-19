
import pandas as pd

# ------------------------------------------
# Q1: Build Personalized Knowledge Base
# ------------------------------------------

fixed_entries = [
    {
        "question": "what is the annual fee",
        "answer": "The annual fee is Rs 500.",
        "keywords": "fee cost price charge",
        "category": "billing"
    },
    {
        "question": "how to reset password",
        "answer": "Go to Settings > Reset Password.",
        "keywords": "password reset login",
        "category": "account"
    },
    {
        "question": "what are your working hours",
        "answer": "We are open from 9 AM to 5 PM.",
        "keywords": "hours timing open time",
        "category": "general"
    },
    {
        "question": "how can i pay the fee",
        "answer": "You can pay via UPI, card, or net banking.",
        "keywords": "pay payment upi fee",
        "category": "billing"
    }
]

# Roll Number = 1024170125
# Last two digits = 2 and 5
# 2 % 3 = 2 -> general
# 5 % 3 = 2 -> general

personal_entries = [
    {
        "question": "how can i contact customer support",
        "answer": "You can contact customer support through email or helpline.",
        "keywords": "support help contact email phone",
        "category": "general"
    },
    {
        "question": "where is your office located",
        "answer": "Our office is located in Chandigarh.",
        "keywords": "office address location map",
        "category": "general"
    }
]

faq_entries = fixed_entries + personal_entries

df = pd.DataFrame(faq_entries)

print("\n========== Q1 ==========")
print(df.to_string(index=False))


# ------------------------------------------
# Q2: Generate and Score a Hypothesis
# ------------------------------------------

def score_query(query, df):
    query_words = query.lower().split()

    scores = []

    for _, row in df.iterrows():

        text = (
            row["question"] + " " +
            row["keywords"] + " " +
            row["category"]
        ).lower()

        score = 0

        for word in query_words:
            if word in text:
                score += 1

        scores.append(score)

    result = df.copy()
    result["score"] = scores

    result = result[result["score"] > 0]
    result = result.sort_values(by="score", ascending=False)

    return result


print("\n========== Q2 ==========")

query = "fee payment"

print("Query:", query)

print(score_query(query, df).to_string(index=False))


# ------------------------------------------
# Q3: Same Category Function
# ------------------------------------------

def same_category(category_name, df):
    return df[df["category"] == category_name]


print("\n========== Q3 ==========")

category = personal_entries[0]["category"]

print("Category:", category)

print(same_category(category, df).to_string(index=False))


# ------------------------------------------
# Q4: Add New Keyword and Save CSV
# ------------------------------------------

print("\n========== Q4 ==========")

entry_index = 0

new_keyword = input("Enter a new keyword: ").lower().strip()

df.at[entry_index, "keywords"] = (
    df.at[entry_index, "keywords"] + " " + new_keyword
)

filename = "1024170125_faq_data.csv"

df.to_csv(filename, index=False)

print("\nUpdated DataFrame:")
print(df.to_string(index=False))

print("\nCSV file saved as:", filename)


# ------------------------------------------
# Q5: GroupBy
# ------------------------------------------

print("\n========== Q5 ==========")

count = df.groupby("category").size()

print(count)


# ------------------------------------------
# Q6: Tie Handling
# ------------------------------------------

def score_query_with_tie(query, df):

    query_words = query.lower().split()

    scores = []

    for _, row in df.iterrows():

        text = (
            row["question"] + " " +
            row["keywords"] + " " +
            row["category"]
        ).lower()

        score = 0

        for word in query_words:
            if word in text:
                score += 1

        scores.append(score)

    temp = df.copy()

    temp["score"] = scores

    highest = temp["score"].max()

    if highest == 0:
        print("No matching FAQ found.")
        return

    best = temp[temp["score"] == highest]

    if len(best) > 1:
        print("\nTie Found! Matching Entries:\n")
    else:
        print("\nBest Match:\n")

    print(best[["question", "answer", "score"]].to_string(index=False))


print("\n========== Q6 ==========")

print("\nExample producing a tie:")

score_query_with_tie("fee", df)

print("\nExample without a tie:")

score_query_with_tie("password", df)
