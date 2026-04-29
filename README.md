Bike Recommendation System

A simple web-based bike recommendation system built using Flask, Pandas, and Scikit-learn.
This app suggests bikes based on user preferences such as description, engine capacity, and price.

📌 Features
🔍 Search bikes using natural language (e.g., "affordable city bike")
⚙️ Filter by:
Engine CC
Price range
🧠 Uses TF-IDF and cosine similarity for text-based matching
📊 Combines:
Text similarity score
Rule-based scoring (engine + price)
🌐 Clean and simple web interface

🛠️ Tech Stack
Backend: Flask (Python)
Frontend: HTML, CSS
Data Processing: Pandas
NLP: NLTK
Machine Learning: Scikit-learn

project/
│── app.py
│── recbike.csv
│── templates/
│   └── index.html
│── README.md

⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/bike-recommendation-system.git
cd bike-recommendation-system

2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install dependencies
pip install flask pandas scikit-learn nltk

▶️ Running the App
python app.py

Then open your browser and go to:
http://127.0.0.1:5000/

1. Text Processing
  Lowercasing
  Removing HTML & special characters
  Tokenization
  Stopword removal
  Stemming
2. Vectorization
  Uses TF-IDF based on user query words
3. Similarity Calculation
  Computes cosine similarity between user query and bike descriptions
4. Scoring System
  Final Score is calculated as:
  Final Score = 0.8 * Similarity Score - 0.2 * Rule Score

Where:
  Similarity Score → Text match quality
  Rule Score → Difference in:
  Engine CC
  Price
  
🖥️ Usage
Enter a description (required)
Optionally enter:
Engine CC (or type cc to ignore)
Price (or type price to ignore)
Click Get Recommendations
View top 5 recommended bikes

📸 Example Query
Description: sporty fast bike
Engine CC: 1000
Price: 1200000

🚀 Future Improvements
    Add image support for bikes
    Improve recommendation algorithm (e.g., hybrid ML model)
    Add user login & saved preferences
    Deploy on cloud (Heroku / AWS)

👨‍💻 Author
Your Name
GitHub: https://github.com/sharad-2006
