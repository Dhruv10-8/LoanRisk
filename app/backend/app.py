from flask import Flask, jsonify, request
import os
import shap
import numpy as np
import xgboost as xgb
import pandas as pd
import psycopg2


app = Flask(__name__)
model = xgb.XGBClassifier()
model.load_model("../../model/xgb_model.json")

def get_db_connection():
    return psycopg2.connect(
        host = os.getenv('DB_HOST'),
        port = os.getenv('DB_PORT', 5432),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    
@app.route("/submit-record", methods=["POST"])
def submit_record():
    data = request.get_json()
    cols = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL',
            'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
            'FLAG_MOBIL','FLAG_PHONE','FLAG_EMAIL','OCCUPATION_TYPE','CNT_FAM_MEMBERS','STATUS','age']

    values = tuple(data[col] for col in cols)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO bankrecords ({','.join(cols)})
        VALUES ({','.join(['%s'] * len(cols))})
        RETURNING account_number
    """, values)
    acc_num = cur.fetchone()[0]
    conn.commit()
    conn.close()

    return jsonify({"message": "Record inserted", "account_number": acc_num})

@app.route("/assess", methods=["POST"])
def assess_risk():
    account_number = request.get_json().get("account_number")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM bankrecords WHERE account_number = %s", (account_number,))
    row = cur.fetchone()
    if not row:
        return jsonify({"error": "Account not found"}), 404

    columns = [desc[0] for desc in cur.description][1:]  # exclude account_number
    conn.close()

    X = pd.DataFrame([row[1:]], columns=columns)
    pred = model.predict(X)[0]
    shap_values = explainer(X)

    # 🔹 Store to loan_assessments
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO loan_assessments (account_number, prediction, shap_values, created_at)
        VALUES (%s, %s, %s, NOW())
    """, (account_number, float(pred), shap_values.values[0].tolist()))

    # 🔹 Update bankrecords with predicted loan_approval
    cur.execute("""
        UPDATE bankrecords SET loan_approval = %s WHERE account_number = %s
    """, (int(pred), account_number))

    conn.commit()
    conn.close()

    return jsonify({
        "prediction": float(pred),
        "shap_values": shap_values.values[0].tolist(),
        "features": X.columns.tolist(),
        "feature_values": X.iloc[0].tolist()
    })

@app.route("/explanation/<int:assessment_id>", methods=["GET"])
def get_explanation(assessment_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT account_number, prediction, shap_values, created_at
        FROM loan_assessments WHERE id = %s
    """, (assessment_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Assessment not found"}), 404

    return jsonify({
        "account_number": row[0],
        "prediction": row[1],
        "shap_values": row[2],
        "created_at": row[3].isoformat()
    })

@app.route("/retrain-if-needed", methods=["POST"])
def retrain_if_needed():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM loan_assessments")
    count = cur.fetchone()[0]

    if count < 10000:
        conn.close()
        return jsonify({"message": f"Only {count} records. Retrain not triggered."})

    cur.execute("""
        SELECT b.*, l.created_at
        FROM bankrecords b
        JOIN loan_assessments l ON l.account_number = b.account_number
        ORDER BY l.created_at DESC
        LIMIT 10000
    """)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]

    conn.close()

    df = pd.DataFrame(rows, columns=cols)

    if "loan_approval" not in df.columns:
        return jsonify({"error": "'loan_approval' column missing in training data"}), 500

    X = df.drop(columns=["account_number", "loan_approval", "created_at"])
    y = df["loan_approval"]

    # 🔁 Prepare DMatrix
    dtrain = xgb.DMatrix(X, label=y)

    booster = xgb.Booster()
    booster.load_model("../../model/xgb_model.json")

    booster.update(dtrain, iteration=0)

    booster.save_model("../../model/xgb_model.json")

    global model, explainer
    model = xgb.XGBClassifier()
    model.load_model("../../model/xgb_model.json")
    explainer = shap.Explainer(model)

    return jsonify({"message": "✅ Model incrementally retrained on latest 10,000 records."})

if __name__ == "__main__":
    app.run(debug=True)