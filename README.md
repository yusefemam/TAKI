 Taki Project

Taki is a web-based Arabic language analysis platform built with Flask. It provides multiple NLP services including text correction, summarization, sentiment analysis, diagnostics, and machine translation. The project is designed to help users process and analyze Arabic text efficiently.

 Features

- **Text Correction:** Automatically corrects linguistic errors in Arabic text.
- **Text Summarization:** Generates concise summaries for input text.
- **Sentiment Analysis:** Detects sentiment (positive/negative/neutral) in Arabic sentences.
- **Diagnostics:** Analyzes text for linguistic and grammatical issues.
- **Machine Translation:** Translates Arabic text to other languages.
- **Email Integration:** Send results via email (if configured).

 Project Structure
main.py
models/
    corrector.py
    diagnostics.py
    summarizer.py
    sentiment.py
    machine_translation.py
    mail.py
data/
    *.csv (datasets for training/testing)
templates/
    chat.html (main UI)
static/
    img/ (images for UI)
logs/
    *.tfevents (logs for model training/events)
results/
    ... (output results)
.idea/
    ... (IDE configs)
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd taki_analysis
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   - Create a `.env` file in the root directory for sensitive configs (API keys, mail settings, etc.).

 Usage

1. **Run the Flask app:**
   ```
   python main.py
   ```
   - The app will start at `http://127.0.0.1:5001/`.

2. **Access the Chat Interface:**
   - Open [http://127.0.0.1:5001/chat_page](http://127.0.0.1:5001/chat_page) in your browser.
   - Select a task, enter your Arabic text, and view results.

 Data

- The `data/` folder contains various CSV datasets for training and evaluation, including:
  - `ANERCorp_Benajiba.csv`
  - `arabic_linguistic_errors_1500.csv`
  - `spam_ham_dataset.csv`
  - `combined_data.csv`
  - ...and more.

 Models

- All NLP models and tools are implemented in the [`models`](models/) directory:
  - [`models/corrector.py`](models/corrector.py): Text correction logic.
  - [`models/summarizer.py`](models/summarizer.py): Summarization logic.
  - [`models/sentiment.py`](models/sentiment.py): Sentiment analysis.
  - [`models/diagnostics.py`](models/diagnostics.py): Diagnostics.
  - [`models/machine_translation.py`](models/machine_translation.py): Translation.
  - [`models/mail.py`](models/mail.py): Email integration.

 UI

- The main user interface is in [`templates/chat.html`](templates/chat.html), styled with Bootstrap and FontAwesome.

 Logging & Results

- Training and event logs are stored in [`logs/`](logs/).
- Output results are saved in [`results/`](results/).

 Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

 License

This project is licensed under the MIT License.

 Contact

For questions or support, please contact:
- **Email:** yousef790326@gmail.com
- **GitHub Issues:** [Open an issue](https://github.com/yusefemam>/<TAKI>/issues)
