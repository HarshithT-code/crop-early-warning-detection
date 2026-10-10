# 🌾 Early Warning System for Crop Yield Loss
### All India | Real-Time Weather | AI Prediction | 64 Crops | 28 States

---

## 📌 Problem Statement

Indian farmers lack access to timely, location-specific early warning information about weather risks and optimal harvest timing — leading to preventable crop yield losses, income instability, and food insecurity across rural India.

> India loses an estimated **₹92,000 crore** worth of crops annually due to weather-related yield losses — most of which are **preventable with early warning**.

---

## 💡 Solution

A real-time, AI-powered web platform that provides Indian farmers with:
- **Personalised crop risk warnings** (Low / Medium / High)
- **Live weather analysis** for their exact district
- **Exact optimal harvest window** based on sowing date
- **Specific action advisory** to prevent yield loss

---

## 🖼️ Features

| Feature | Description |
|---|---|
| 🗺️ All India Coverage | 28 States, 400+ Districts |
| 🌾 64 Crops | Cereals, Pulses, Oilseeds, Vegetables, Fruits, Spices, Plantation |
| 🌡️ Live Weather | Real-time data from Open-Meteo API |
| 🤖 AI Risk Prediction | Random Forest Classifier — Low / Medium / High |
| 📅 Harvest Window | Exact date range based on sowing date + crop growth period |
| 📋 Action Advisory | Specific steps to prevent yield loss per risk level |
| 📱 Responsive | Works on mobile and desktop |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | Random Forest Classifier (scikit-learn) |
| Weather API | Open-Meteo (free, real-time, no key needed) |
| Geocoding | OpenStreetMap Nominatim (free) |
| Frontend | HTML5, CSS3, JavaScript |
| Deployment | Render (free tier) |

---

## 🌱 Crops Supported

**Cereals:** Rice, Wheat, Maize, Ragi, Jowar, Bajra, Barley, Oats, Sorghum

**Pulses:** Chickpea, Pigeon Pea, Moong, Urad, Lentil, Soybean, Cowpea, Field Pea

**Oilseeds:** Groundnut, Mustard, Sunflower, Sesame, Linseed, Castor, Safflower

**Cash/Fibre:** Sugarcane, Cotton, Jute, Tobacco

**Vegetables:** Tomato, Potato, Onion, Chilli, Brinjal, Cabbage, Cauliflower, Okra, Cucumber, Pumpkin, Bitter Gourd, Bottle Gourd, Spinach, Garlic, Ginger, Turmeric

**Fruits:** Mango, Banana, Papaya, Watermelon, Grapes, Pomegranate, Coconut, Guava

**Spices & Plantation:** Coriander, Cumin, Fenugreek, Black Pepper, Cardamom, Tea, Coffee, Rubber, Areca Nut

---

## 🎯 SDG Alignment

| SDG | Goal |
|---|---|
| **SDG 2 ★** | Zero Hunger — prevents crop yield loss |
| **SDG 1** | No Poverty — protects farmer income |
| **SDG 13** | Climate Action — weather-based climate adaptation |
| **SDG 9** | Innovation — AI for rural agriculture |

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/crop-early-warning-detection.git
cd crop-early-warning-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
crop-early-warning-detection/
├── app.py                  ← Flask backend + ML model + routes
├── dataset.py              ← Synthetic dataset generator
├── requirements.txt        ← Python dependencies
├── Procfile                ← Render deployment config
├── templates/
│   ├── base.html           ← Layout (navbar + footer)
│   ├── index.html          ← Home page with input form
│   └── result.html         ← Early warning result page
└── static/css/
    └── style.css           ← Stylesheet
```

---

## 👨‍🌾 Target Users

- Small and marginal farmers (landholding < 2 hectares)
- Agricultural extension officers (KVK workers)
- State agriculture departments
- Agri-NGOs working with rural farmers

---

## 📊 How It Works

```
Farmer selects Crop + State + District + Sowing Date
                        ↓
System geocodes district → fetches live weather (Open-Meteo)
                        ↓
AI model predicts risk: Low / Medium / High
                        ↓
Crop conditions checked vs ideal agronomic thresholds
                        ↓
Exact harvest window calculated from sowing date
                        ↓
Personalised action advisory generated instantly
```

---

*Built to empower Indian farmers with real-time AI advisory — All India Coverage*
