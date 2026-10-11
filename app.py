"""
Early Warning System for Crop Yield Loss
Flask backend: ML prediction, real weather (Open-Meteo), real geocoding (OSM Nominatim)
Run with: python app.py  →  open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import requests
import datetime
from sklearn.ensemble import RandomForestClassifier
from functools import lru_cache

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
#  CROP KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────
CROP_DB = {
    # ── CEREALS ──
    "Rice (Paddy)":            {"growth_days": 120, "sow_months": [6,7],      "ideal_temp": (20,35), "ideal_rain": (150,300), "ideal_hum": (60,90),  "icon": "🌾"},
    "Wheat":                   {"growth_days": 120, "sow_months": [11,12],    "ideal_temp": (12,25), "ideal_rain": (30,100),  "ideal_hum": (40,70),  "icon": "🌿"},
    "Maize":                   {"growth_days": 95,  "sow_months": [6,7],      "ideal_temp": (21,35), "ideal_rain": (60,150),  "ideal_hum": (50,80),  "icon": "🌽"},
    "Ragi (Finger Millet)":    {"growth_days": 105, "sow_months": [6,7],      "ideal_temp": (20,35), "ideal_rain": (50,100),  "ideal_hum": (40,75),  "icon": "🌱"},
    "Jowar (Sorghum)":         {"growth_days": 100, "sow_months": [6,7],      "ideal_temp": (25,38), "ideal_rain": (40,100),  "ideal_hum": (30,65),  "icon": "🌾"},
    "Bajra (Pearl Millet)":    {"growth_days": 75,  "sow_months": [6,7],      "ideal_temp": (25,38), "ideal_rain": (30,90),   "ideal_hum": (30,60),  "icon": "🌾"},
    "Barley":                  {"growth_days": 110, "sow_months": [11,12],    "ideal_temp": (10,25), "ideal_rain": (25,75),   "ideal_hum": (35,65),  "icon": "🌿"},
    "Oats":                    {"growth_days": 100, "sow_months": [11,12],    "ideal_temp": (10,22), "ideal_rain": (30,80),   "ideal_hum": (40,70),  "icon": "🌿"},
    "Sorghum":                 {"growth_days": 95,  "sow_months": [6,7],      "ideal_temp": (25,38), "ideal_rain": (40,100),  "ideal_hum": (30,65),  "icon": "🌾"},

    # ── PULSES ──
    "Chickpea (Chana)":        {"growth_days": 100, "sow_months": [10,11],    "ideal_temp": (15,28), "ideal_rain": (25,75),   "ideal_hum": (30,60),  "icon": "🫘"},
    "Pigeon Pea (Tur/Arhar)":  {"growth_days": 150, "sow_months": [6,7],      "ideal_temp": (20,35), "ideal_rain": (60,150),  "ideal_hum": (40,70),  "icon": "🫘"},
    "Moong (Green Gram)":      {"growth_days": 65,  "sow_months": [6,7,3],    "ideal_temp": (25,35), "ideal_rain": (40,100),  "ideal_hum": (40,75),  "icon": "🫘"},
    "Urad (Black Gram)":       {"growth_days": 70,  "sow_months": [6,7],      "ideal_temp": (25,35), "ideal_rain": (50,100),  "ideal_hum": (40,75),  "icon": "🫘"},
    "Lentil (Masoor)":         {"growth_days": 100, "sow_months": [10,11],    "ideal_temp": (15,25), "ideal_rain": (25,60),   "ideal_hum": (35,65),  "icon": "🫘"},
    "Soybean":                 {"growth_days": 100, "sow_months": [6,7],      "ideal_temp": (20,32), "ideal_rain": (60,130),  "ideal_hum": (50,80),  "icon": "🫘"},
    "Cowpea":                  {"growth_days": 70,  "sow_months": [6,7],      "ideal_temp": (22,35), "ideal_rain": (40,100),  "ideal_hum": (40,75),  "icon": "🫘"},
    "Field Pea":               {"growth_days": 90,  "sow_months": [10,11],    "ideal_temp": (10,22), "ideal_rain": (30,80),   "ideal_hum": (40,70),  "icon": "🫘"},

    # ── OILSEEDS ──
    "Groundnut":               {"growth_days": 110, "sow_months": [6,7],      "ideal_temp": (22,35), "ideal_rain": (50,120),  "ideal_hum": (45,75),  "icon": "🥜"},
    "Mustard":                 {"growth_days": 110, "sow_months": [10,11],    "ideal_temp": (10,25), "ideal_rain": (25,75),   "ideal_hum": (35,65),  "icon": "🌼"},
    "Sunflower":               {"growth_days": 90,  "sow_months": [6,7,11],   "ideal_temp": (18,35), "ideal_rain": (40,100),  "ideal_hum": (35,65),  "icon": "🌻"},
    "Sesame (Til)":            {"growth_days": 80,  "sow_months": [6,7],      "ideal_temp": (25,38), "ideal_rain": (40,100),  "ideal_hum": (35,65),  "icon": "🌱"},
    "Linseed (Flaxseed)":      {"growth_days": 100, "sow_months": [10,11],    "ideal_temp": (10,22), "ideal_rain": (25,60),   "ideal_hum": (35,65),  "icon": "🌼"},
    "Castor":                  {"growth_days": 180, "sow_months": [6,7],      "ideal_temp": (20,35), "ideal_rain": (50,120),  "ideal_hum": (40,70),  "icon": "🌱"},
    "Safflower":               {"growth_days": 120, "sow_months": [10,11],    "ideal_temp": (15,30), "ideal_rain": (25,60),   "ideal_hum": (30,60),  "icon": "🌼"},

    # ── CASH / FIBRE CROPS ──
    "Sugarcane":               {"growth_days": 365, "sow_months": [2,3],      "ideal_temp": (20,38), "ideal_rain": (100,200), "ideal_hum": (50,80),  "icon": "🎋"},
    "Cotton":                  {"growth_days": 180, "sow_months": [5,6],      "ideal_temp": (21,38), "ideal_rain": (50,100),  "ideal_hum": (40,70),  "icon": "🤍"},
    "Jute":                    {"growth_days": 120, "sow_months": [4,5],      "ideal_temp": (24,38), "ideal_rain": (150,300), "ideal_hum": (60,90),  "icon": "🌿"},
    "Tobacco":                 {"growth_days": 90,  "sow_months": [11,12],    "ideal_temp": (18,32), "ideal_rain": (50,120),  "ideal_hum": (45,75),  "icon": "🌿"},

    # ── VEGETABLES ──
    "Tomato":                  {"growth_days": 75,  "sow_months": [1,7,10],   "ideal_temp": (18,30), "ideal_rain": (40,100),  "ideal_hum": (45,70),  "icon": "🍅"},
    "Potato":                  {"growth_days": 90,  "sow_months": [10,11],    "ideal_temp": (10,24), "ideal_rain": (40,90),   "ideal_hum": (45,80),  "icon": "🥔"},
    "Onion":                   {"growth_days": 120, "sow_months": [10,11],    "ideal_temp": (13,28), "ideal_rain": (30,80),   "ideal_hum": (40,70),  "icon": "🧅"},
    "Chilli":                  {"growth_days": 90,  "sow_months": [6,7,11],   "ideal_temp": (20,32), "ideal_rain": (40,100),  "ideal_hum": (50,75),  "icon": "🌶️"},
    "Brinjal (Eggplant)":      {"growth_days": 80,  "sow_months": [6,7,11],   "ideal_temp": (22,32), "ideal_rain": (40,100),  "ideal_hum": (45,75),  "icon": "🍆"},
    "Cabbage":                 {"growth_days": 85,  "sow_months": [9,10],     "ideal_temp": (10,22), "ideal_rain": (40,100),  "ideal_hum": (45,75),  "icon": "🥬"},
    "Cauliflower":             {"growth_days": 80,  "sow_months": [9,10],     "ideal_temp": (10,22), "ideal_rain": (40,90),   "ideal_hum": (45,75),  "icon": "🥦"},
    "Okra (Bhindi)":           {"growth_days": 55,  "sow_months": [6,7,3],    "ideal_temp": (25,38), "ideal_rain": (40,100),  "ideal_hum": (40,75),  "icon": "🌿"},
    "Cucumber":                {"growth_days": 55,  "sow_months": [6,7,3],    "ideal_temp": (20,32), "ideal_rain": (40,100),  "ideal_hum": (50,80),  "icon": "🥒"},
    "Pumpkin":                 {"growth_days": 90,  "sow_months": [6,7],      "ideal_temp": (20,35), "ideal_rain": (50,120),  "ideal_hum": (45,75),  "icon": "🎃"},
    "Bitter Gourd (Karela)":   {"growth_days": 60,  "sow_months": [6,7,3],    "ideal_temp": (25,38), "ideal_rain": (40,100),  "ideal_hum": (45,75),  "icon": "🌿"},
    "Bottle Gourd (Lauki)":    {"growth_days": 60,  "sow_months": [6,7,3],    "ideal_temp": (25,38), "ideal_rain": (40,100),  "ideal_hum": (45,75),  "icon": "🌿"},
    "Spinach":                 {"growth_days": 45,  "sow_months": [10,11,2],  "ideal_temp": (10,24), "ideal_rain": (30,80),   "ideal_hum": (40,70),  "icon": "🥬"},
    "Garlic":                  {"growth_days": 130, "sow_months": [10,11],    "ideal_temp": (12,25), "ideal_rain": (25,75),   "ideal_hum": (35,65),  "icon": "🧄"},
    "Ginger":                  {"growth_days": 210, "sow_months": [4,5],      "ideal_temp": (20,30), "ideal_rain": (100,200), "ideal_hum": (60,85),  "icon": "🌿"},
    "Turmeric":                {"growth_days": 240, "sow_months": [5,6],      "ideal_temp": (20,35), "ideal_rain": (100,200), "ideal_hum": (60,85),  "icon": "🌿"},

    # ── FRUITS ──
    "Mango":                   {"growth_days": 120, "sow_months": [3,4],      "ideal_temp": (24,38), "ideal_rain": (60,150),  "ideal_hum": (40,70),  "icon": "🥭"},
    "Banana":                  {"growth_days": 300, "sow_months": [3,4,9],    "ideal_temp": (20,35), "ideal_rain": (100,200), "ideal_hum": (60,85),  "icon": "🍌"},
    "Papaya":                  {"growth_days": 270, "sow_months": [6,7,3],    "ideal_temp": (22,36), "ideal_rain": (60,150),  "ideal_hum": (50,80),  "icon": "🍈"},
    "Watermelon":              {"growth_days": 80,  "sow_months": [3,4],      "ideal_temp": (22,35), "ideal_rain": (40,100),  "ideal_hum": (40,70),  "icon": "🍉"},
    "Grapes":                  {"growth_days": 150, "sow_months": [1,2],      "ideal_temp": (18,35), "ideal_rain": (40,100),  "ideal_hum": (35,70),  "icon": "🍇"},
    "Pomegranate":             {"growth_days": 180, "sow_months": [7,8],      "ideal_temp": (18,35), "ideal_rain": (40,100),  "ideal_hum": (35,65),  "icon": "🍎"},
    "Coconut":                 {"growth_days": 365, "sow_months": [5,6],      "ideal_temp": (22,35), "ideal_rain": (100,200), "ideal_hum": (60,90),  "icon": "🥥"},
    "Guava":                   {"growth_days": 240, "sow_months": [7,8],      "ideal_temp": (20,36), "ideal_rain": (60,150),  "ideal_hum": (45,75),  "icon": "🍏"},

    # ── SPICES & PLANTATION ──
    "Coriander (Dhania)":      {"growth_days": 60,  "sow_months": [10,11],    "ideal_temp": (15,28), "ideal_rain": (25,75),   "ideal_hum": (35,65),  "icon": "🌿"},
    "Cumin (Jeera)":           {"growth_days": 90,  "sow_months": [10,11],    "ideal_temp": (12,26), "ideal_rain": (20,60),   "ideal_hum": (30,60),  "icon": "🌿"},
    "Fenugreek (Methi)":       {"growth_days": 75,  "sow_months": [10,11],    "ideal_temp": (12,28), "ideal_rain": (25,70),   "ideal_hum": (35,65),  "icon": "🌿"},
    "Black Pepper":            {"growth_days": 365, "sow_months": [5,6],      "ideal_temp": (20,30), "ideal_rain": (150,250), "ideal_hum": (60,90),  "icon": "🌿"},
    "Cardamom":                {"growth_days": 365, "sow_months": [5,6],      "ideal_temp": (15,25), "ideal_rain": (150,300), "ideal_hum": (70,90),  "icon": "🌿"},
    "Tea":                     {"growth_days": 365, "sow_months": [3,4],      "ideal_temp": (18,28), "ideal_rain": (150,300), "ideal_hum": (70,90),  "icon": "🍃"},
    "Coffee":                  {"growth_days": 365, "sow_months": [5,6],      "ideal_temp": (18,28), "ideal_rain": (150,250), "ideal_hum": (65,85),  "icon": "☕"},
    "Rubber":                  {"growth_days": 365, "sow_months": [5,6],      "ideal_temp": (22,32), "ideal_rain": (200,350), "ideal_hum": (70,90),  "icon": "🌳"},
    "Areca Nut (Supari)":      {"growth_days": 365, "sow_months": [5,6],      "ideal_temp": (22,32), "ideal_rain": (150,300), "ideal_hum": (60,90),  "icon": "🌴"},
}

INDIA_STATES = {
    "Andhra Pradesh":   ["Anantapur","Chittoor","East Godavari","Guntur","Krishna","Kurnool","Nellore","Prakasam","Srikakulam","Visakhapatnam","Vizianagaram","West Godavari","YSR Kadapa"],
    "Assam":            ["Barpeta","Bongaigaon","Cachar","Darrang","Dhemaji","Dibrugarh","Goalpara","Jorhat","Kamrup","Lakhimpur","Nagaon","Sivasagar","Sonitpur","Tinsukia"],
    "Bihar":            ["Araria","Aurangabad","Bhagalpur","Buxar","Darbhanga","Gaya","Muzaffarpur","Nalanda","Patna","Purnia","Rohtas","Samastipur","Saran","Vaishali","West Champaran"],
    "Chhattisgarh":     ["Bastar","Bilaspur","Durg","Janjgir-Champa","Korba","Raigarh","Raipur","Rajnandgaon","Surguja"],
    "Goa":              ["North Goa","South Goa"],
    "Gujarat":          ["Ahmedabad","Amreli","Anand","Banaskantha","Bharuch","Bhavnagar","Gandhinagar","Jamnagar","Junagadh","Kutch","Mehsana","Rajkot","Surat","Surendranagar","Vadodara"],
    "Haryana":          ["Ambala","Bhiwani","Faridabad","Fatehabad","Gurugram","Hisar","Jhajjar","Jind","Karnal","Kurukshetra","Mahendragarh","Panchkula","Panipat","Rewari","Rohtak","Sirsa","Sonipat"],
    "Himachal Pradesh": ["Bilaspur","Chamba","Hamirpur","Kangra","Kinnaur","Kullu","Mandi","Shimla","Sirmaur","Solan","Una"],
    "Jharkhand":        ["Bokaro","Deoghar","Dhanbad","East Singhbhum","Giridih","Hazaribagh","Khunti","Koderma","Latehar","Ranchi","Sahebganj","Seraikela"],
    "Karnataka":        ["Bagalkot","Ballari","Bangalore Rural","Bangalore Urban","Belagavi","Bidar","Chamarajanagar","Chikkaballapura","Chikkamagaluru","Chitradurga","Dakshina Kannada","Davangere","Dharwad","Gadag","Hassan","Haveri","Kalaburagi","Kodagu","Kolar","Koppal","Mandya","Mysuru","Raichur","Ramanagara","Shivamogga","Tumakuru","Udupi","Uttara Kannada"],
    "Kerala":           ["Alappuzha","Ernakulam","Idukki","Kannur","Kasaragod","Kollam","Kottayam","Kozhikode","Malappuram","Palakkad","Pathanamthitta","Thiruvananthapuram","Thrissur","Wayanad"],
    "Madhya Pradesh":   ["Betul","Bhopal","Chhindwara","Dewas","Dhar","Gwalior","Indore","Jabalpur","Katni","Khandwa","Mandla","Morena","Narsinghpur","Rewa","Sagar","Satna","Shahdol","Shivpuri","Ujjain","Vidisha"],
    "Maharashtra":      ["Ahmednagar","Akola","Amravati","Aurangabad","Beed","Bhandara","Buldhana","Chandrapur","Dhule","Gadchiroli","Gondia","Jalgaon","Jalna","Kolhapur","Latur","Mumbai","Nagpur","Nanded","Nashik","Osmanabad","Palghar","Parbhani","Pune","Raigad","Ratnagiri","Sangli","Satara","Solapur","Thane","Wardha","Washim","Yavatmal"],
    "Manipur":          ["Bishnupur","Churachandpur","Imphal East","Imphal West","Senapati","Thoubal","Ukhrul"],
    "Meghalaya":        ["East Garo Hills","East Khasi Hills","Ri Bhoi","South Garo Hills","West Garo Hills","West Khasi Hills"],
    "Mizoram":          ["Aizawl","Champhai","Kolasib","Lunglei","Saiha","Serchhip"],
    "Nagaland":         ["Dimapur","Kohima","Mokokchung","Mon","Phek","Tuensang","Wokha","Zunheboto"],
    "Odisha":           ["Angul","Balangir","Balasore","Bargarh","Bhadrak","Cuttack","Deogarh","Dhenkanal","Gajapati","Ganjam","Jagatsinghpur","Jajpur","Jharsuguda","Kalahandi","Keonjhar","Khordha","Koraput","Mayurbhanj","Nayagarh","Puri","Rayagada","Sambalpur","Sundargarh"],
    "Punjab":           ["Amritsar","Barnala","Bathinda","Faridkot","Fatehgarh Sahib","Fazilka","Ferozepur","Gurdaspur","Hoshiarpur","Jalandhar","Kapurthala","Ludhiana","Mansa","Moga","Pathankot","Patiala","Rupnagar","Sangrur","Tarn Taran"],
    "Rajasthan":        ["Ajmer","Alwar","Banswara","Barmer","Bharatpur","Bhilwara","Bikaner","Bundi","Chittorgarh","Dausa","Dungarpur","Jaipur","Jaisalmer","Jalore","Jodhpur","Kota","Nagaur","Pali","Pratapgarh","Rajsamand","Sikar","Sirohi","Sri Ganganagar","Tonk","Udaipur"],
    "Sikkim":           ["East Sikkim","North Sikkim","South Sikkim","West Sikkim"],
    "Tamil Nadu":       ["Ariyalur","Chennai","Coimbatore","Cuddalore","Dharmapuri","Dindigul","Erode","Kanchipuram","Kanyakumari","Karur","Krishnagiri","Madurai","Nagapattinam","Namakkal","Nilgiris","Perambalur","Pudukkottai","Ramanathapuram","Salem","Sivaganga","Thanjavur","Theni","Thoothukudi","Tiruchirappalli","Tirunelveli","Tiruvallur","Tiruvannamalai","Vellore","Viluppuram","Virudhunagar"],
    "Telangana":        ["Adilabad","Bhadradri Kothagudem","Hyderabad","Jagtial","Karimnagar","Khammam","Mahabubnagar","Mancherial","Medak","Medchal","Nalgonda","Nizamabad","Peddapalli","Rangareddy","Sangareddy","Siddipet","Suryapet","Vikarabad","Warangal","Wanaparthy","Yadadri"],
    "Tripura":          ["Dhalai","Gomati","Khowai","North Tripura","Sepahijala","South Tripura","Unakoti","West Tripura"],
    "Uttar Pradesh":    ["Agra","Aligarh","Allahabad","Ambedkar Nagar","Azamgarh","Bahraich","Ballia","Bareilly","Basti","Bijnor","Bulandshahr","Chandauli","Deoria","Etah","Etawah","Farrukhabad","Fatehpur","Firozabad","Gautam Buddha Nagar","Ghaziabad","Ghazipur","Gonda","Gorakhpur","Hardoi","Jaunpur","Jhansi","Kanpur","Kaushambi","Kushinagar","Lakhimpur Kheri","Lucknow","Maharajganj","Mainpuri","Mathura","Mau","Meerut","Mirzapur","Moradabad","Muzaffarnagar","Pilibhit","Pratapgarh","Rae Bareli","Rampur","Saharanpur","Shahjahanpur","Sitapur","Sonbhadra","Sultanpur","Unnao","Varanasi"],
    "Uttarakhand":      ["Almora","Bageshwar","Chamoli","Champawat","Dehradun","Haridwar","Nainital","Pauri Garhwal","Pithoragarh","Rudraprayag","Tehri Garhwal","Udham Singh Nagar","Uttarkashi"],
    "West Bengal":      ["Alipurduar","Bankura","Birbhum","Cooch Behar","Darjeeling","Hooghly","Howrah","Jalpaiguri","Jhargram","Kalimpong","Kolkata","Malda","Murshidabad","Nadia","North 24 Parganas","Paschim Bardhaman","Paschim Medinipur","Purba Bardhaman","Purba Medinipur","Purulia","South 24 Parganas","Uttar Dinajpur"],
}

# Dynamic advisory is now generated by build_advisory() below — no static dict needed.

# ─────────────────────────────────────────────────────────────
#  ML MODEL — trained once at startup
# ─────────────────────────────────────────────────────────────
def _build_model():
    np.random.seed(42)
    n = 10_000
    rain = np.random.randint(0, 500, n).astype(float)
    temp = np.random.randint(10, 45,  n).astype(float)
    hum  = np.random.randint(20, 100, n).astype(float)
    ndvi = np.round(np.random.uniform(0.1, 0.9, n), 2)
    pest = np.random.randint(0, 100, n).astype(float)
    labels = np.array([
        2 if sum([rain[i]<30 or rain[i]>350, temp[i]>38 or temp[i]<12,
                  hum[i]>85, ndvi[i]<0.3, pest[i]>70]) >= 3
        else 1 if sum([rain[i]<30 or rain[i]>350, temp[i]>38 or temp[i]<12,
                       hum[i]>85, ndvi[i]<0.3, pest[i]>70]) >= 1
        else 0
        for i in range(n)
    ])
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(np.column_stack([rain, temp, hum, ndvi, pest]), labels)
    return clf

MODEL = _build_model()

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=512)
def geocode(district: str, state: str):
    try:
        url = (f"https://nominatim.openstreetmap.org/search"
               f"?q={district}+{state}+India&format=json&limit=1")
        res = requests.get(url, headers={"User-Agent": "KrishiRakshak-EWS/2.0"}, timeout=10).json()
        if res:
            return float(res[0]["lat"]), float(res[0]["lon"])
    except Exception:
        pass
    return 20.59, 78.96

def fetch_weather(lat: float, lon: float) -> dict:
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
            f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
            f"&timezone=Asia%2FKolkata&forecast_days=7"
        )
        data  = requests.get(url, timeout=12).json()
        cur   = data.get("current", {})
        daily = data.get("daily", {})
        return {
            "temp":            cur.get("temperature_2m", 28.0),
            "humidity":        cur.get("relative_humidity_2m", 65),
            "rain_now":        cur.get("precipitation", 0.0),
            "wind":            cur.get("wind_speed_10m", 8.0),
            "weather_code":    cur.get("weather_code", 0),
            "forecast_rain_7d": round(sum(daily.get("precipitation_sum", [0]*7)), 1),
            "temp_max_7d":     max(daily.get("temperature_2m_max", [28]*7)),
            "temp_min_7d":     min(daily.get("temperature_2m_min", [18]*7)),
        }
    except Exception:
        return {"temp":28,"humidity":65,"rain_now":0,"wind":8,"weather_code":0,
                "forecast_rain_7d":10,"temp_max_7d":32,"temp_min_7d":20}

def weather_label(code: int) -> str:
    if code == 0:  return "Clear sky"
    if code <= 3:  return "Partly cloudy"
    if code <= 9:  return "Foggy"
    if code <= 29: return "Rain / Drizzle"
    if code <= 39: return "Snow"
    if code <= 69: return "Heavy rain"
    if code <= 84: return "Rain showers"
    if code <= 99: return "Thunderstorm"
    return "Unknown"

def ndvi_proxy(rain: float, temp: float) -> float:
    if 50 <= rain <= 200 and 18 <= temp <= 35: return 0.65
    if rain > 200 or temp > 38:               return 0.45
    if rain < 20:                              return 0.20
    return 0.35

def run_prediction(weather: dict, crop_key: str) -> tuple:
    """Crop-aware risk scoring: compares real weather against THIS crop's ideal ranges."""
    info         = CROP_DB[crop_key]
    monthly_rain = weather["rain_now"] + weather["forecast_rain_7d"] * 4.3
    temp         = weather["temp"]
    hum          = weather["humidity"]
    ndvi         = ndvi_proxy(monthly_rain, temp)

    t_min, t_max = info["ideal_temp"]
    h_min, h_max = info["ideal_hum"]
    r_min, r_max = info["ideal_rain"]

    # Score each stress factor against crop-specific thresholds
    stress_flags = [
        monthly_rain < r_min or monthly_rain > r_max,   # rainfall out of range for THIS crop
        temp < t_min or temp > t_max,                   # temperature out of range for THIS crop
        hum  < h_min or hum  > h_max,                  # humidity out of range for THIS crop
        ndvi < 0.3,                                     # poor vegetation index
        hum > 85 and temp > (t_max - 5),               # disease-pressure: high hum near heat limit
    ]
    stress_count = sum(stress_flags)

    if   stress_count >= 3: risk = 2
    elif stress_count >= 1: risk = 1
    else:                   risk = 0

    # Confidence: percentage of thresholds that are "ok" (inverted stress ratio)
    proba = round((1 - stress_count / len(stress_flags)) * 100, 1)

    return risk, proba, ndvi, round(monthly_rain, 1)

def harvest_window(crop_key: str, sowing_str: str):
    info  = CROP_DB[crop_key]
    sow   = datetime.date.fromisoformat(sowing_str)
    base  = sow + datetime.timedelta(days=info["growth_days"])
    start = base - datetime.timedelta(days=7)
    end   = base + datetime.timedelta(days=7)
    today = datetime.date.today()
    days_left = (start - today).days
    return start, end, days_left

def condition_check(weather: dict, crop_key: str) -> list:
    info         = CROP_DB[crop_key]
    temp         = weather["temp"]
    hum          = weather["humidity"]
    monthly_rain = weather["rain_now"] + weather["forecast_rain_7d"] * 4.3
    results      = []

    t_min, t_max = info["ideal_temp"]
    if   temp < t_min: results.append(("Temperature", "low",  f"{temp}°C — Too cold for {crop_key} (ideal {t_min}–{t_max}°C)"))
    elif temp > t_max: results.append(("Temperature", "high", f"{temp}°C — Too hot for {crop_key} (ideal {t_min}–{t_max}°C)"))
    else:              results.append(("Temperature", "ok",   f"{temp}°C — Within ideal range for {crop_key} ({t_min}–{t_max}°C)"))

    h_min, h_max = info["ideal_hum"]
    if   hum < h_min: results.append(("Humidity", "low",  f"{hum}% — Too dry for {crop_key} (ideal {h_min}–{h_max}%)"))
    elif hum > h_max: results.append(("Humidity", "high", f"{hum}% — Too humid for {crop_key} (ideal {h_min}–{h_max}%)"))
    else:             results.append(("Humidity", "ok",   f"{hum}% — Within ideal range for {crop_key} ({h_min}–{h_max}%)"))

    r_min, r_max = info["ideal_rain"]
    if   monthly_rain < r_min: results.append(("Rainfall", "low",  f"~{monthly_rain:.0f} mm/mo — Too low for {crop_key} (ideal {r_min}–{r_max} mm/mo)"))
    elif monthly_rain > r_max: results.append(("Rainfall", "high", f"~{monthly_rain:.0f} mm/mo — Excess for {crop_key} (ideal {r_min}–{r_max} mm/mo)"))
    else:                      results.append(("Rainfall", "ok",   f"~{monthly_rain:.0f} mm/mo — Adequate for {crop_key} ({r_min}–{r_max} mm/mo)"))

    return results


def build_advisory(risk: int, weather: dict, crop_key: str, conditions: list) -> list:
    """Generate advisory messages specific to this crop, location and actual failing conditions."""
    info  = CROP_DB[crop_key]
    temp  = weather["temp"]
    hum   = weather["humidity"]
    wind  = weather["wind"]
    rain7 = weather["forecast_rain_7d"]

    # Map condition statuses for quick lookup
    cond_map = {param: status for param, status, _ in conditions}
    temp_status = cond_map.get("Temperature", "ok")
    hum_status  = cond_map.get("Humidity",    "ok")
    rain_status = cond_map.get("Rainfall",    "ok")

    msgs = []

    if risk == 0:
        msgs.append(("✅", f"Conditions are favourable for {crop_key}. Proceed with normal field operations."))
        msgs.append(("💊", f"Continue routine pest scouting every 3–4 days as a preventive measure."))
        msgs.append(("🌱", f"Apply the recommended fertiliser dose suitable for this growth stage of {crop_key}."))
        msgs.append(("📅",  "Prepare post-harvest storage and grading equipment in advance."))
        return msgs

    # --- MEDIUM / HIGH: condition-specific messages ---
    if risk == 2:
        msgs.append(("🔴", f"HIGH RISK for {crop_key} in current conditions — immediate action required."))

    # Temperature stress
    if temp_status == "high":
        t_max = info["ideal_temp"][1]
        msgs.append(("🌡️", f"Temperature {temp}°C exceeds {crop_key}'s limit ({t_max}°C). Apply foliar spray with water early morning to reduce heat stress."))
        msgs.append(("🌿",  "Mulch the field to retain soil moisture and lower root-zone temperature."))
    elif temp_status == "low":
        t_min = info["ideal_temp"][0]
        msgs.append(("🥶", f"Temperature {temp}°C is below {crop_key}'s minimum ({t_min}°C). Use crop covers or polythene mulch to protect from cold stress."))

    # Humidity stress
    if hum_status == "high":
        msgs.append(("🍄", f"Humidity {hum}% is too high for {crop_key} — risk of fungal diseases. Apply Copper Oxychloride (3 g/L) or Mancozeb fungicide."))
        msgs.append(("💨",  "Ensure adequate plant spacing and canopy airflow to reduce moisture retention."))
    elif hum_status == "low":
        msgs.append(("💧", f"Humidity {hum}% is too low for {crop_key}. Increase irrigation frequency; consider drip irrigation to maintain soil moisture."))

    # Rainfall stress
    if rain_status == "high":
        msgs.append(("🚿",  "Stop irrigation immediately — excess rainfall forecast."))
        msgs.append(("🏗️", f"Create emergency drainage channels. Standing water can cause root rot in {crop_key}."))
        if risk == 2:
            msgs.append(("📸",  "Document crop damage with photos for insurance claim under PM Fasal Bima Yojana."))
    elif rain_status == "low":
        msgs.append(("🪣", f"Low rainfall forecast ({rain7} mm/7 days). Schedule irrigation every 3–4 days for {crop_key}."))
        msgs.append(("🌾",  "Apply organic mulch to reduce water evaporation from soil surface."))

    # Wind stress
    if wind > 40:
        msgs.append(("💨", f"Wind speed {wind} km/h is high. Stake tall {crop_key} plants to prevent lodging."))

    # Pest pressure from high humidity + heat
    if hum > 80 and temp > info["ideal_temp"][1] - 5:
        msgs.append(("🐛", f"High humidity and heat increases pest pressure for {crop_key}. Deploy pheromone traps and apply Neem oil (3 ml/L)."))

    # Always add escalation for HIGH risk
    if risk == 2:
        msgs.append(("📞",  "Contact your District Agriculture Officer or Krishi Vigyan Kendra (KVK) immediately."))
    else:
        msgs.append(("📞",  "Increase field monitoring to every 2 days. Contact your local KVK if conditions worsen."))

    return msgs


YIELD_LOSS = {0: 8, 1: 28, 2: 65}
RISK_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
RISK_TAG   = {0: "safe", 1: "warning", 2: "danger"}

# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        states=sorted(INDIA_STATES.keys()),
        crops=list(CROP_DB.keys()),
        crop_db=CROP_DB,
        india_states=INDIA_STATES,
        today=datetime.date.today().isoformat(),
        min_date=(datetime.date.today() - datetime.timedelta(days=365)).isoformat(),
        max_date=(datetime.date.today() + datetime.timedelta(days=180)).isoformat(),
    )

@app.route("/districts")
def districts():
    state = request.args.get("state", "")
    return jsonify(sorted(INDIA_STATES.get(state, [])))

@app.route("/predict", methods=["POST"])
def predict():
    crop_key    = request.form.get("crop")
    state       = request.form.get("state")
    district    = request.form.get("district")
    sowing_date = request.form.get("sowing_date")

    if not all([crop_key, state, district, sowing_date]):
        return render_template("index.html",
            states=sorted(INDIA_STATES.keys()), crops=list(CROP_DB.keys()),
            crop_db=CROP_DB, india_states=INDIA_STATES,
            today=datetime.date.today().isoformat(),
            min_date=(datetime.date.today()-datetime.timedelta(days=365)).isoformat(),
            max_date=(datetime.date.today()+datetime.timedelta(days=180)).isoformat(),
            error="Please fill all fields before submitting.")

    lat, lon    = geocode(district, state)
    weather     = fetch_weather(lat, lon)
    risk, proba, ndvi, monthly_rain = run_prediction(weather, crop_key)
    h_start, h_end, days_left = harvest_window(crop_key, sowing_date)
    conditions  = condition_check(weather, crop_key)
    advisory    = build_advisory(risk, weather, crop_key, conditions)
    crop_info   = CROP_DB[crop_key]

    return render_template(
        "result.html",
        # inputs
        crop_key    = crop_key,
        crop_info   = crop_info,
        state       = state,
        district    = district,
        sowing_date = datetime.date.fromisoformat(sowing_date).strftime("%d %b %Y"),
        # location
        lat         = round(lat, 4),
        lon         = round(lon, 4),
        # weather
        weather     = weather,
        weather_sky = weather_label(weather["weather_code"]),
        # prediction
        risk        = risk,
        risk_label  = RISK_LABEL[risk],
        risk_tag    = RISK_TAG[risk],
        proba       = proba,
        yield_loss  = YIELD_LOSS[risk],
        ndvi        = round(ndvi, 2),
        # harvest
        h_start     = h_start.strftime("%d %b %Y"),
        h_end       = h_end.strftime("%d %b %Y"),
        days_left   = days_left,
        growth_days = crop_info["growth_days"],
        # analysis
        conditions  = conditions,
        advisory    = advisory,
        # meta
        generated   = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p IST"),
    )

if __name__ == "__main__":
    app.run(debug=True)
