🏆 Gold Price Predictor & Investment Advisor
Welcome to the Gold Price Predictor — an AI-powered app that analyzes gold market trends and gives intelligent investment recommendations, powered by Machine Learning and Gemini AI!

📊 Project Overview
This project predicts short-term gold price movements using advanced technical indicators, machine learning models, and real-time AI insights.
It provides BUY or WAIT recommendations based on:

Historical gold price trends

Technical analysis features (Moving Averages, RSI, Bollinger Bands, etc.)

Machine Learning (Gradient Boosting, Random Forest Classifiers)

Real-time AI market insights using Google Gemini API

The app is deployed via Streamlit for an interactive user experience.

🚀 Live Demo
👉 Click here to view the app!
(Replace with your real Streamlit link after hosting.)

🛠️ Features
📈 Fetches real-time gold market data (via Yahoo Finance API)

🔍 Creates 25+ technical indicators automatically

🧠 Trains Machine Learning models (Gradient Boosting + Random Forest)

📊 Visualizes prediction confidence and feature importances

🤖 Integrates Gemini AI for global market sentiment analysis

🌐 Supports USD to INR conversion using real-time exchange rates

📉 Displays historical trends and "Buy Signals"

💬 Gives final actionable investment advice (Strong Buy, Wait, or Mixed)

🏗️ Tech Stack
Python 3.10+

Streamlit (web app UI)

Pandas, NumPy (data manipulation)

Scikit-Learn (machine learning)

Yahoo Finance API (yfinance) (live financial data)

Google Gemini AI API (market insights)

Matplotlib, Seaborn (visualization)

Dotenv (managing secrets safely)

🧩 Project Structure
bash
Copy
Edit
gold-price-predictor/
├── gold_predictor.py      # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable sample
├── README.md              # Project documentation
└── (venv/)                # (Local virtual environment - not pushed)
⚙️ How to Run Locally
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/gold-price-predictor.git
cd gold-price-predictor
2. Set up a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
3. Install required packages
bash
Copy
Edit
pip install -r requirements.txt
4. Set up your .env file
Create a .env file based on .env.example:

ini
Copy
Edit
GEMINI_API_KEY=your_actual_gemini_api_key_here
(Make sure you have a valid Google Gemini API Key.)

5. Run the Streamlit app
bash
Copy
Edit
streamlit run gold_predictor.py
🛡️ Deployment on Streamlit Cloud
Push this repo to GitHub.

Go to Streamlit Cloud ➔ New App ➔ Connect your GitHub repo.

Set your secrets (GEMINI_API_KEY) in the app settings.

Click Deploy 🚀

📈 Example Screenshots
Add screenshots of your app here once deployed (optional but highly recommended)

🤝 Contributing
Contributions are welcome! Feel free to open issues or pull requests if you'd like to improve this project.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
Yahoo Finance API

Google Gemini API

Streamlit

