from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "Flask backend is running 🚀"


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # === Step 1: Read uploaded file ===
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # === Step 2: Standardize columns ===
        df.columns = [c.strip() for c in df.columns]
        df.rename(
            columns={
                "Months": "Month",
                "Cash Inflow": "Inflow",
                "Cash Outflow": "Outflow",
                "Net Cash Flow": "Net_Cashflow",
            },
            inplace=True,
        )

        print("🧾 Raw month values (first 5):", df["Month"].head().tolist())

        # === Step 3: Parse months ===
        df["Month"] = pd.to_datetime(df["Month"], format="%y-%b", errors="coerce")
        if df["Month"].isna().all():
            df["Month"] = pd.to_datetime(df["Month"], format="%d-%b-%y", errors="coerce")

        df = df.dropna(subset=["Month"]).sort_values("Month")

        # === Step 4: Convert numerics ===
        df["Inflow"] = pd.to_numeric(df["Inflow"], errors="coerce")
        df["Outflow"] = pd.to_numeric(df["Outflow"], errors="coerce")
        df["Net_Cashflow"] = df["Inflow"] + df["Outflow"]
        df = df.dropna(subset=["Inflow", "Outflow"])

        if len(df) < 3:
            return jsonify({"error": "Not enough valid rows for forecasting"}), 400

        # === Step 5: Forecast Inflow and Outflow separately ===
        def safe_forecast(series, label):
            """Forecast a numeric series with exponential smoothing."""
            try:
                model = ExponentialSmoothing(series, trend="add", seasonal=None)
                fit = model.fit(optimized=True)
                return fit.forecast(12)
            except Exception as e:
                print(f"⚠️ Fallback SimpleExponentialSmoothing for {label}: {e}")
                fit = SimpleExpSmoothing(series).fit()
                return fit.forecast(12)

        inflow_series = pd.Series(df["Inflow"].values, index=df["Month"])
        outflow_series = pd.Series(df["Outflow"].values, index=df["Month"])

        inflow_forecast = safe_forecast(inflow_series, "Inflow")
        outflow_forecast = safe_forecast(outflow_series, "Outflow")

        # === Step 6: Build forecast dataframe ===
        last_date = df["Month"].iloc[-1]
        forecast_index = pd.date_range(
            start=pd.Timestamp(last_date) + pd.offsets.MonthBegin(1),
            periods=12,
            freq="MS",
        )

        forecast_df = pd.DataFrame({
            "Month": forecast_index.strftime("%b-%Y"),
            "Inflow_Forecast": np.round(inflow_forecast.values, 2),
            "Outflow_Forecast": np.round(outflow_forecast.values, 2),
        })

        # Compute Net Cashflow forecast as sum of inflow + outflow
        forecast_df["Net_Cashflow_Forecast"] = (
            forecast_df["Inflow_Forecast"] + forecast_df["Outflow_Forecast"]
        )

        # === Step 7: Combine with history ===
        df_display = df.copy()
        df_display["Month"] = df_display["Month"].dt.strftime("%b-%Y")

        # Match all columns
        forecast_df["Inflow"] = forecast_df["Inflow_Forecast"]
        forecast_df["Outflow"] = forecast_df["Outflow_Forecast"]
        forecast_df["Net_Cashflow"] = forecast_df["Net_Cashflow_Forecast"]

        # Same column order
        combined = pd.concat([df_display, forecast_df], ignore_index=True)
        combined = combined.replace([np.nan, np.inf, -np.inf], None)
        combined = combined.dropna(how="all")

        print("\n✅ Final Combined Data (last few rows):\n", combined.tail())

        # === Step 8: Return to frontend ===
        return jsonify({
            "status": "success",
            "rows_received": len(df),
            "forecast": combined.to_dict(orient="records"),
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
