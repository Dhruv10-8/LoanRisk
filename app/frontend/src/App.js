// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LabelList } from 'recharts';

function App() {
  const initialForm = {
    CODE_GENDER: 1,
    FLAG_OWN_CAR: 0,
    FLAG_OWN_REALTY: 1,
    CNT_CHILDREN: 0,
    AMT_INCOME_TOTAL: 50000,
    NAME_INCOME_TYPE: 0,
    NAME_EDUCATION_TYPE: 1,
    NAME_FAMILY_STATUS: 2,
    NAME_HOUSING_TYPE: 1,
    FLAG_MOBIL: 1,
    FLAG_PHONE: 1,
    FLAG_EMAIL: 1,
    OCCUPATION_TYPE: 3,
    CNT_FAM_MEMBERS: 3,
    STATUS: 1,
    age: 30
  };

  const [formData, setFormData] = useState(initialForm);
  const [accountNumber, setAccountNumber] = useState(null);
  const [inputAcc, setInputAcc] = useState('');
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: isNaN(value) ? value : Number(value) }));
  };

  const handleSubmit = async () => {
    try {
      const res = await axios.post('/submit-record', formData);
      setAccountNumber(res.data.account_number);
      alert("✅ Record inserted!");
    } catch (err) {
      alert("❌ Failed to insert: " + err.message);
    }
  };

  const handleAssess = async () => {
    try {
      const res = await axios.post('/assess', { account_number: Number(inputAcc) });
      setResult(res.data);
    } catch (err) {
      alert("❌ Error: " + err.response?.data?.error || err.message);
    }
  };

  const shapData = result?.features.map((feature, idx) => ({
    name: feature,
    shap: parseFloat(result.shap_values[idx]),
    value: result.feature_values[idx]
  })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

  return (
    <div style={{ maxWidth: "900px", margin: "auto", padding: "1rem" }}>
      <h2>📝 Submit Bank Record</h2>
      {Object.entries(formData).map(([key, val]) => (
        <div key={key} style={{ marginBottom: '10px' }}>
          <label>{key}</label>
          <input name={key} value={val} onChange={handleChange} style={{ marginLeft: 10 }} />
        </div>
      ))}
      <button onClick={handleSubmit}>Submit Record</button>
      {accountNumber && <p>✅ Account inserted with number: {accountNumber}</p>}

      <hr />

      <h2>🔍 Assess Risk</h2>
      <input
        placeholder="Enter account number"
        value={inputAcc}
        onChange={e => setInputAcc(e.target.value)}
        style={{ marginRight: 10 }}
      />
      <button onClick={handleAssess}>Assess</button>

      {result && (
        <>
          <h3>Prediction: {result.prediction === 1 ? "Loan Approved ✅" : "Rejected ❌"}</h3>
          <h4>Top Feature Impacts:</h4>
          <div style={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={shapData.slice(0, 10)}>
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={150} />
                <Tooltip />
                <Bar dataKey="shap" fill="#8884d8">
                  <LabelList dataKey="value" position="right" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
