# 🤖 <a href = https://rnn-project-by-muhammad-umar.streamlit.app>RNN Based Chatbot with Adjustable Metrics</a>
  
This project is a **Shakespeare-Hamlet inspired chatbot** built using a **Recurrent Neural Network (RNN)** trained on text data.  
It is deployed using **Streamlit** and allows users to fine-tune the text generation behavior with adjustable parameters such as **temperature**, **Top-K sampling**, and **Top-P (nucleus) sampling**.  

---

## 🚀 Features
- **Interactive Chat Interface** powered by Streamlit.
- **Adjustable Sampling Parameters**:
  - 🔥 **Temperature**: Controls randomness in predictions.  
    - Lower values → more deterministic.  
    - Higher values → more creative but risky.  
  - 🔢 **Top-K Sampling**: Restricts the model to the top K probable words.  
  - 📊 **Top-P (Nucleus Sampling)**: Chooses words from the smallest set of tokens whose probabilities add up to P.  
- **Conversation History** is maintained during the session.
- Trained on **Hamlet by William Shakespeare**.

---

## 📂 Project Structure

```

.
├── models/
│   └── best\_model\_acc.keras       # Trained RNN model
├── pipelines/
│   └── tokenizer.pkl              # Tokenizer for text sequences
├── app.py                         # Main Streamlit app
├── data/                          # (Optional) Text dataset
├── README.md                      # Project Documentation

````



## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rnn-chatbot.git
   cd rnn-chatbot
    ```

2. Create a virtual environment and activate it:

   ``` bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ``` bash
   pip install -r requirements.txt
   ```

---

## 📦 Requirements

* Python **3.13**
* TensorFlow (latest, supports Python 3.13)
* Streamlit
* NumPy
* Pickle (built-in with Python)

Create a `requirements.txt` file with:

```txt
tensorflow>=2.17.0
streamlit>=1.38.0
numpy>=1.26.0
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

* Enter a **prompt** from Hamlet or your own text.
* Adjust **Temperature**, **Top-K**, and **Top-P** from the sidebar.
* The assistant will generate text based on your input.

---

## 📜 Example

**User Input**:

```
To be or not to be
```

**Model Output** (with Temperature = 1.0, Top-K = 5, Top-P = 0.5):

```
To be or not to be the question of the soul
```

---

## 📌 Notes

* Model was trained on **Shakespeare's Hamlet**.
* Outputs are **not always accurate**, but mimic Shakespearean style.
* Adjusting **temperature, Top-K, and Top-P** can drastically change responses.

---

## 📜 License

This project is licensed under the **MIT License** – you’re free to use, modify, and distribute it.

---

