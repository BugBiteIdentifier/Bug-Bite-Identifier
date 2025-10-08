# This file is for testing different the results page with different bug type inputs quickly
# Each time you want to change bug input you must refresh page, which will then send another prompt in the terminal
from flask import Flask, render_template
import time

app = Flask(__name__)

@app.route("/", endpoint="home")
def index():
    ##enter case to test in terminal
    predicted_label = input("Please enter your label (ant, bedbug, tick, berry bug, spider, fleas, mosquito, or no bite)\n")  
    return render_template("results.html", label=predicted_label)


if __name__ == "__main__":
    app.run(debug=True)
    # page would not properly take input and load without an amount of time spent waiting
    time.sleep(10)
