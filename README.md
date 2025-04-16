# 🎓 Advisify: Thesis Adviser Recommender System

**Advisify** is a thesis adviser recommendation system built using Streamlit and Sentence Transformers. It helps students match with the most suitable thesis advisers based on the similarity between their proposed thesis titles and the expertise of available advisers.

## 🚀 Features

- 🔍 **Semantic Similarity Matching** using precomputed embeddings and cosine similarity
- 🧠 **Dual Matching Logic**:
  - Adviser's **Area of Expertise**
  - Adviser's **Past Thesis Topics Supervised**
- 🎯 **Weighted Scoring System** for balanced recommendation
- 🛠️ **Optional Project Type Selection** for more accurate results
- 📊 **Visual Similarity Scores** via progress bars
- ⚡ Fast and scalable inference using pre-generated embeddings

## 📁 Project Structure

```
├── advisers_data.csv          # Contains adviser info + precomputed embeddings
├── app.py                     # Main Streamlit app file
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🧑‍💻 How It Works

1. Users input a detailed thesis title.
2. (Optional) Users can select general project types to aid matching.
3. The app encodes the input using `sentence-transformers/all-MiniLM-L6-v2`.
4. It computes the cosine similarity against:
   - Adviser’s **expertise embeddings**
   - Adviser’s **past thesis embeddings**
5. The overall match score is computed using a weighted formula:
   ```
   final_score = 0.6 * expertise_similarity + 0.4 * past_thesis_similarity
   ```
6. The top 5 adviser matches are displayed.

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/advisify.git
   cd advisify
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🧾 Requirements

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `sentence-transformers`

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## 📚 Dataset

The file `advisers_data.csv` must contain the following columns:

- `name`: Adviser name
- `area_of_expertise_description`
- `past_thesis_topics_supervised`
- `expertise_embeddings`: Precomputed embedding vector as JSON string
- `past_thesis_embeddings`: Precomputed embedding vector as JSON string

## 🛡️ Notes

- Embeddings are precomputed for better performance.
- App is designed to work **offline** after setup.
- For best results, advise students to input **clear and specific** thesis titles.

## 💡 Future Improvements

- Admin dashboard to add/update adviser data
- User login and personalized history
- Integration with actual thesis management systems
- Feedback loop to improve matching quality


---

📬 For questions or feedback, feel free to reach out!
