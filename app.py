from flask import Flask, request, render_template
from model import train_and_predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None
    error = None
    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper().strip()
        if symbol:
            try:
                predicted_price = train_and_predict(symbol)
                predicted_price = f"${predicted_price:.2f}"
            except Exception as e:
                error = f"Could not predict price for '{symbol}'. Error: {e}"
        else:
            error = "Please enter a valid stock ticker symbol."
    return render_template("index.html", predicted=predicted_price, error=error)

if __name__ == "__main__":
    app.run(debug=True)
