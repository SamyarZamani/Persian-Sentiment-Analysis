# House Price Prediction

## 📌 Project Overview
This project aims to **predict house prices** in Iran based on various features such as area, number of rooms, parking availability, warehouse, elevator, and address. The model uses **Random Forest Regressor** to estimate house prices both in Iranian Rial and USD.

---

## 🗂 Dataset
- **Source:** `housePrice.csv`  
- **Number of entries:** 3,479  
- **Columns:**
  - `Area` (numeric) – size of the house in square meters  
  - `Room` (integer) – number of rooms  
  - `Parking` (boolean) – parking availability  
  - `Warehouse` (boolean) – warehouse availability  
  - `Elevator` (boolean) – elevator availability  
  - `Address` (categorical) – neighborhood or location  
  - `Price` (float) – price in Rial  
  - `Price(USD)` (float) – price in USD  

**Missing values:**  
- `Address` had 23 missing values, filled with `"Unknown"`  
- `Area` had 6 missing values, which were removed  

---

## ⚙ Preprocessing
1. Fill missing `Address` values and encode using `LabelEncoder`  
2. Convert `Area` to numeric  
3. Convert boolean features (`Parking`, `Warehouse`, `Elevator`) to integers  
4. Scale all features using `StandardScaler`  
5. Split dataset into **train** and **test** sets (80% / 20%)  

---

## 🧠 Model
- **Algorithm:** Random Forest Regressor  
- **Parameters:**  
  - `n_estimators=100`  
  - `max_depth=15`  
  - `random_state=42`  
- Separate models for:
  - **Price in Rial**  
  - **Price in USD**

---

## 📊 Model Evaluation

| Target      | MSE                  | RMSE            | R² Score |
|------------|--------------------|----------------|----------|
| Price (Rial) | 15,075,203,087,057,416,192 | 3,882,679,885 | 0.780    |
| Price (USD)  | 16,547,629,234                | 128,638        | 0.782    |

The R² score indicates that the model can explain around 78% of the variance in house prices.

---

## 🏠 Predicting New Values
Example:  
```python
area = 150
room = 3
parking = 1
warehouse = 1
elevator = 1
address_encoded = 2

new_data = np.array([[area, room, parking, warehouse, elevator, address_encoded]])
new_data_scaled = scaler.transform(new_data)

predicted_price_rial = model_rial.predict(new_data_scaled)
predicted_price_usd = model_usd.predict(new_data_scaled)

print(predicted_price_rial)  # 8,337,453,333 Rial
print(predicted_price_usd)   # 274,843 USD
## 🔧 Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib (optional, for visualization)

---

## 🚀 Usage
1. Upload `housePrice.csv`
2. Run preprocessing and training cells
3. Use trained models to predict house prices

---

## 💡 Notes & Recommendations
- Label encoding for `Address` works, but **One-Hot Encoding** may improve accuracy
- Hyperparameter tuning (using `GridSearchCV`) can improve model performance
- Consider using advanced models like **XGBoost** or **LightGBM** for better predictions

