import pandas as pd
import numpy as np

FILE_PATH = "data/Baltalimani_Data_2025.xlsx"
OUTPUT_PATH = "data/cleaned_data.xlsx"


def load_data():
    df = pd.read_excel(FILE_PATH)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df


def convert_times(df):
    time_columns = [
        "GIRIS_TARIHI",
        "RANDEVU_BASLAMA_SAATI",
        "MUAYENE_KABUL_ZAMANI",
        "CAGRILMA_ZAMANI",
        "TETKIK_ISTEK_SAATI",
        "CEKIM_ZAMANI",
        "MUAYENE_SONLANDIRMA_ZAMANI",
    ]

    print("\n--- TIME CONVERSION ---")
    for col in time_columns:
        if col in df.columns:
            before_null = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            after_null = df[col].isna().sum()
            print(f"{col}: before_null={before_null}, after_null={after_null}, new_parse_errors={after_null - before_null}")
    return df


def create_features(df):
    print("\n--- FEATURE CREATION ---")

    # Whether patient needed X-ray
    df["HAS_XRAY"] = (
        df["TETKIK_ISTEK_SAATI"].notna() &
        df["CEKIM_ZAMANI"].notna()
    ).astype(int)

    # Exam duration: called -> exam finished
    df["EXAM_DURATION_MIN"] = (
        df["MUAYENE_SONLANDIRMA_ZAMANI"] - df["CAGRILMA_ZAMANI"]
    ).dt.total_seconds() / 60

    # X-ray flow time: request -> X-ray taken
    # Important: this is NOT pure service time; it includes waiting in the X-ray stage
    df["XRAY_FLOW_MIN"] = (
        df["CEKIM_ZAMANI"] - df["TETKIK_ISTEK_SAATI"]
    ).dt.total_seconds() / 60

    # Appointment delay: scheduled -> accepted
    if "RANDEVU_BASLAMA_SAATI" in df.columns and "MUAYENE_KABUL_ZAMANI" in df.columns:
        df["APPOINTMENT_DELAY_MIN"] = (
            df["MUAYENE_KABUL_ZAMANI"] - df["RANDEVU_BASLAMA_SAATI"]
        ).dt.total_seconds() / 60

    # Clinic / doctor helper fields
    df["DOKTOR_ADI"] = df["DOKTOR_ADI"].astype(str).str.strip()

    print("Created columns: HAS_XRAY, EXAM_DURATION_MIN, XRAY_FLOW_MIN, APPOINTMENT_DELAY_MIN")
    return df


def clean_data(df):
    print("\n--- CLEANING DATA ---")
    initial_rows = len(df)

    # Keep rows with usable exam duration
    df = df[df["CAGRILMA_ZAMANI"].notna()]
    df = df[df["MUAYENE_SONLANDIRMA_ZAMANI"].notna()]
    df = df[df["EXAM_DURATION_MIN"].notna()]
    df = df[df["EXAM_DURATION_MIN"] >= 0]
    df = df[df["EXAM_DURATION_MIN"] <= 180]

    # Clean appointment delay if present, but do not drop rows just because appointment data is missing
    if "APPOINTMENT_DELAY_MIN" in df.columns:
        invalid_appt = (df["APPOINTMENT_DELAY_MIN"] < -180) | (df["APPOINTMENT_DELAY_MIN"] > 180)
        df.loc[invalid_appt, "APPOINTMENT_DELAY_MIN"] = np.nan

    # Clean X-ray flow only for patients who have X-ray
    has_xray_mask = df["HAS_XRAY"] == 1
    invalid_xray = has_xray_mask & (
        df["XRAY_FLOW_MIN"].isna() |
        (df["XRAY_FLOW_MIN"] < 0) |
        (df["XRAY_FLOW_MIN"] > 180)
    )

    # If X-ray timestamps are inconsistent, mark them as non-usable X-ray records
    df.loc[invalid_xray, "HAS_XRAY"] = 0
    df.loc[invalid_xray, "XRAY_FLOW_MIN"] = np.nan
    df.loc[invalid_xray, "TETKIK_ISTEK_SAATI"] = pd.NaT
    df.loc[invalid_xray, "CEKIM_ZAMANI"] = pd.NaT

    final_rows = len(df)

    print(f"Initial rows: {initial_rows}")
    print(f"Remaining rows: {final_rows}")
    print(f"Removed rows: {initial_rows - final_rows}")
    print(f"X-ray probability after cleaning: {df['HAS_XRAY'].mean():.4f}")

    return df


def print_summary(df):
    print("\n--- FINAL SUMMARY ---")
    print("\nExam duration summary:")
    print(df["EXAM_DURATION_MIN"].describe())

    if df["HAS_XRAY"].sum() > 0:
        print("\nX-ray flow summary (for X-ray patients only):")
        print(df.loc[df["HAS_XRAY"] == 1, "XRAY_FLOW_MIN"].describe())

    print("\nPatients with X-ray:", int(df["HAS_XRAY"].sum()))
    print("Patients without X-ray:", int((df["HAS_XRAY"] == 0).sum()))
    print("Unique doctors:", df["DOKTOR_ADI"].nunique())


def main():
    print("=== START DATA PROCESSING ===")
    df = load_data()
    df = convert_times(df)
    df = create_features(df)
    df = clean_data(df)
    print_summary(df)

    df.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()