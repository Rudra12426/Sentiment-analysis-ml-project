# Step 1 : import libraries 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 2: Movie Dataset
reviews = [
    "This movie was amazing",
    "I loved the film",
    "This movie was fantastic",
    "The film was terrible",
    "I hated this movie",
    "This was the worst movie"
]

labels = [1,1,1,0,0,0]
# 1 =Positive 0 : Negative
# for example ,amazing = positive , worst = negative

# Step 3 : Text processing function 
def preprocess(text):
# Lower Case
  text= text.lower()
# toeknization using split
  tokens =text.split()
# Remove Puctuation
  tokens = [word.strip(string.punctuation) for word in tokens ]
# Joins word back
  text =" ".join(tokens)
  return text

# Step 4: Clean aLl Reviews 
clean_reviews = [preprocess(review) for review in reviews]

# Step 5 : Convert text to numbers 
vectorizer=TfidfVectorizer()
X = vectorizer.fit_transform(clean_reviews)

# Step 6 : Train the models 
model = LogisticRegression()
model.fit(X, labels)

# Step 7 : Test the model
test_review=["The movie was amazing"]
test_clean=[preprocess(r) for r in test_review]
test_vector=vectorizer.transform(test_clean)
prediction=model.predict(test_vector)

# Step 8 : Print Result
if prediction[0] == 1:
 print("Positive Review")
else:
 print("Negative Review")
