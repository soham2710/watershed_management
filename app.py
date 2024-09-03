from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
import pymongo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import plotly.express as px
from pymongo import MongoClient
import certifi
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


# Use your actual connection string
connection_string = "mongodb+srv://sohamnsharma:rdcv4c75@watershed.xbd1e.mongodb.net/?retryWrites=true&w=majority&appName=Watershed"
client = MongoClient(connection_string, tlsCAFile=certifi.where())
db = client['watershed_management']
collection = db['watershed_db']
user_db = client['user_management']
user_collection = user_db['users']

model = None
preprocessor = None

def load_data():
    documents = collection.find()
    df = pd.DataFrame(documents)
    df['combined'] = df['st_name'] + '_' + df['dist_name']
    collection.update_many({}, {"$set": {"combined": {"$concat": ["$st_name", "_", "$dist_name"]}}})
    return df

def preprocess_data(df):
    global preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('scaler', StandardScaler())]), ['total_project', 'project_area']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['combined'])
        ]
    )
    X = df[['combined', 'total_project', 'project_area']]
    y = df['project_cost']
    if X.empty or y.empty:
        raise ValueError("No data available for processing. Please check the data source or filtering criteria.")
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def cache_model():
    global model, preprocessor
    df = load_data()
    X_train, y_train = preprocess_data(df)
    model = train_model(X_train, y_train)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

def load_cached_model():
    global model, preprocessor
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        if user_collection.find_one({'username': username}):
            flash('Username already exists, please choose a different one.')
            return redirect(url_for('signup'))
        user_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Signup successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = user_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful!')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

def login_required(f):
    def wrap(*args, **kwargs):
        if 'username' not in session:
            flash('You need to log in first.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

def plot_data(df, state, city):
    city_data = df[(df['st_name'] == state) & (df['dist_name'] == city)]
    fig = px.scatter(city_data, x='project_area', y='project_cost', title=f'Project Area vs. Cost for {city}, {state}',
                     labels={'project_area': 'Project Area (Hectares)', 'project_cost': 'Project Cost (Lakhs)'})
    return fig

def plot_state_data(df, state):
    state_data = df[df['st_name'] == state]
    fig = px.scatter(state_data, x='project_area', y='project_cost', color='dist_name', title=f'Project Area vs. Cost for {state}',
                     labels={'project_area': 'Project Area (Hectares)', 'project_cost': 'Project Cost (Lakhs)'})
    return fig


@app.route('/')
@login_required
def index():
    df = load_data()
    total_projects = len(df)
    total_area = df['project_area'].sum()
    total_cost = df['project_cost'].sum()
    states = df['st_name'].unique().tolist()
    state_costs = df.groupby('st_name')['project_cost'].sum().to_dict()
    state_areas = df.groupby('st_name')['project_area'].sum().to_dict()
    cost_chart = px.bar(df.groupby('st_name')['project_cost'].sum().reset_index(), x='st_name', y='project_cost', title='State-wise Distribution of Cost')
    area_chart = px.bar(df.groupby('st_name')['project_area'].sum().reset_index(), x='st_name', y='project_area', title='State-wise Project Area Allocation')
    scatter_plot = px.scatter(df, x='project_area', y='project_cost', color='st_name', title='2D Scatter Plot of Project Data')
    return render_template('index.html', total_projects=total_projects, total_area=total_area, total_cost=total_cost, states=states, state_costs=state_costs, state_areas=state_areas, cost_chart=cost_chart.to_json(), area_chart=area_chart.to_json(), scatter_plot=scatter_plot.to_json())

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    df = load_data()
    states = df['st_name'].unique().tolist()
    districts = df['dist_name'].unique().tolist()
    predicted_cost = None
    city_plot_html = None
    state_plot_html = None

    if model is None:
        load_cached_model()

    if request.method == 'POST':
        state = request.form.get('state')
        city = request.form.get('city')
        project_area = float(request.form.get('project_area'))
        total_projects = len(df[(df['st_name'] == state) & (df['dist_name'] == city)])
        input_data = pd.DataFrame({'st_name': [state], 'dist_name': [city], 'total_project': [total_projects], 'project_area': [project_area]})
        input_data['combined'] = input_data['st_name'] + '_' + input_data['dist_name']
        X_input = preprocessor.transform(input_data)
        predicted_cost = model.predict(X_input)[0]
        city_fig = plot_data(df, state, city)
        state_fig = plot_state_data(df, state)
        city_plot_html = city_fig.to_html(full_html=False)
        state_plot_html = state_fig.to_html(full_html=False)

    return render_template('predict.html', states=states, districts=districts, predicted_cost=predicted_cost, city_plot_html=city_plot_html, state_plot_html=state_plot_html)


@app.route('/get_districts')
@login_required
def get_districts():
    state = request.args.get('state')
    districts = collection.distinct("dist_name", {"st_name": state})
    return jsonify(districts)

if __name__ == '__main__':
    cache_model()
    app.run(debug=False)

