from flask import Flask, redirect, request, url_for, render_template, flash
from model import get_user_list, get_top_5_recommendations

app = Flask(__name__)
app.config["SECRET_KEY"] = "development"

## Load user list
user_list = get_user_list()

@app.route("/")
async def index():
    return render_template("index.html", user_list=user_list, user=None, show_recommendations=False, recommend_products=list())

@app.route("/recommend", methods=['GET', 'POST'])
async def recommend():
    ## Get username
    user = request.form.get("user_name")

    if user==None or user=="--SELECT--":
        flash("Please select a user from the dropdown list", "danger")
        return redirect(url_for("index"))
    
    ## Get model recommendations
    recommend_products = get_top_5_recommendations(user)
    
    return render_template("index.html", user_list=user_list, user=user, show_recommendations=True, recommend_products=recommend_products)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0")

