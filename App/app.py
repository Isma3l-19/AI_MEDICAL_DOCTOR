from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Sample training data
# Disease symptoms

disease_data = {
    "Flu" : [
        "fever cough headache body pain chills",
        "high temperature sore throat cough"
    ],
    "Common Cold" : [
        "sneezing runny nose mild fever",
        "cough congestion cold"
    ],
    "Migraine" : [
        "sever headache nausea sensitivity to light",
        "head pain vomiting migraine"
    ],
    "Food Poisoning" : [
        "vomiting diarrhea stomach pain",
        "nausea loose motion abdominal pain"
    ],
    "Allergy" : [
        "itching skin rash sneezing",
        "watery eyes swelling allergy"
    ]
}

# Medical advice for diseases (example data)
medicine_advice = {
    "Flu": {
        "medicines" : [
            "Paracetamol (500mg)",
            "Ibuprofen (200mg)"
        ],
        "dose": "1 tablet every 6-8 hours",
        "time": "Morning, Afternoon, Night"
    },
    "Common Cold": {
        "medicines" : [
            "Cetirizine (10mg)",
            "Loratadine (10mg)"
        ],
        "dose": "1 tablet daily",
        "time": "Morning"
    },
    "Migraine": {
        "medicines" : [
            "Sumatriptan (50mg)",
            "Ibuprofen (400mg)"
        ],
        "dose": "1 tablet at onset of symptoms",
        "time": "As needed"
    },
    "Food Poisoning": {
        "medicines" : [
            "Ondansetron (4mg)",
            "Loperamide (2mg)"
        ],
        "dose": "1 tablet every 8 hours",
        "time": "Morning, Afternoon, Night"
    },
    "Allergy": {
        "medicines" : [
            "Cetirizine (10mg)",
            "Loratadine (10mg)"
        ],
        "dose": "1 tablet daily",
        "time": "Morning"
    }
}

# Training the model

texts, labels = [], []
for disease, symptoms in disease_data.items():
    for symptom in symptoms:
        texts.append(symptom)
        labels.append(disease)

vectorier = CountVectorizer()
x = vectorier.fit_transform(texts)

model = MultinomialNB()
model.fit(x, labels)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get("symptoms").lower()
    vect = vectorier.transform([symptoms])
    disease = model.predict(vect)[0]

    info = medicine_advice.get(disease, {})
    return jsonify({
        "condition": disease,
        "medicines": info.get("medicines", []),
        "dose": info.get("dose", ""),
        "time": info.get("time", ""),
        "disclaimer": "This is a basic AI model and may not be accurate for all cases. Please consult a healthcare professional for proper diagnosis and treatment."
    })

# Run the Application
if __name__ == '__main__':
    app.run(debug=True)