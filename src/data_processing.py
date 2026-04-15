import pandas as pd
import numpy as np

FILE_PATH = "data/Baltalimani_Data_2025.xlsx"
OUTPUT_PATH = "data/cleaned_data.xlsx"
DOCTOR_PARAMS_PATH = "data/doctor_params.xlsx"


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

    df["DOKTOR_ADI"] = df["DOKTOR_ADI"].astype(str).str.strip()

    df["HAS_XRAY"] = (
        df["TETKIK_ISTEK_SAATI"].notna() &
        df["CEKIM_ZAMANI"].notna()
    ).astype(int)

    df["EXAM_DURATION_MIN"] = (
        df["MUAYENE_SONLANDIRMA_ZAMANI"] - df["CAGRILMA_ZAMANI"]
    ).dt.total_seconds() / 60

    df["XRAY_FLOW_MIN"] = (
        df["CEKIM_ZAMANI"] - df["TETKIK_ISTEK_SAATI"]
    ).dt.total_seconds() / 60

    if "RANDEVU_BASLAMA_SAATI" in df.columns and "MUAYENE_KABUL_ZAMANI" in df.columns:
        df["APPOINTMENT_DELAY_MIN"] = (
            df["MUAYENE_KABUL_ZAMANI"] - df["RANDEVU_BASLAMA_SAATI"]
        ).dt.total_seconds() / 60

    print("Created columns: HAS_XRAY, EXAM_DURATION_MIN, XRAY_FLOW_MIN, APPOINTMENT_DELAY_MIN")
    return df


def clean_data(df):
    print("\n--- CLEANING DATA ---")
    initial_rows = len(df)

    # doctor must exist
    df = df[df["DOKTOR_ADI"].notna()]
    df = df[df["DOKTOR_ADI"].astype(str).str.strip() != ""]

    # main exam data must exist
    df = df[df["CAGRILMA_ZAMANI"].notna()]
    df = df[df["MUAYENE_SONLANDIRMA_ZAMANI"].notna()]
    df = df[df["EXAM_DURATION_MIN"].notna()]
    df = df[df["EXAM_DURATION_MIN"] >= 0]
    df = df[df["EXAM_DURATION_MIN"] <= 180]

    # clean appointment delay softly
    if "APPOINTMENT_DELAY_MIN" in df.columns:
        invalid_appt = (df["APPOINTMENT_DELAY_MIN"] < -180) | (df["APPOINTMENT_DELAY_MIN"] > 180)
        df.loc[invalid_appt, "APPOINTMENT_DELAY_MIN"] = np.nan

    # clean xray only when patient has xray
    has_xray_mask = df["HAS_XRAY"] == 1
    invalid_xray = has_xray_mask & (
        df["XRAY_FLOW_MIN"].isna() |
        (df["XRAY_FLOW_MIN"] < 0) |
        (df["XRAY_FLOW_MIN"] > 180)
    )

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


def create_doctor_params(df):
    print("\n--- DOCTOR PARAMETER TABLE ---")

    doctor_stats = df.groupby("DOKTOR_ADI").agg(
        patient_count=("DOKTOR_ADI", "size"),
        exam_mean=("EXAM_DURATION_MIN", "mean"),
        exam_median=("EXAM_DURATION_MIN", "median"),
        xray_prob=("HAS_XRAY", "mean"),
        xray_flow_mean=("XRAY_FLOW_MIN", "mean"),
    ).reset_index()

    # use only doctors with enough observations
    doctor_stats = doctor_stats[doctor_stats["patient_count"] >= 20].copy()

    # X-ray service proxy to avoid double counting
    doctor_stats["xray_service_mean"] = (
        doctor_stats["xray_flow_mean"].fillna(8.0).clip(lower=2.0) * 0.4
    ).clip(upper=12.0)

    doctor_stats["exam_mean"] = doctor_stats["exam_mean"].clip(lower=1.0)
    doctor_stats["xray_prob"] = doctor_stats["xray_prob"].fillna(0.0).clip(lower=0.0, upper=1.0)

    doctor_stats = doctor_stats.sort_values("patient_count", ascending=False)

    print("Doctors kept in parameter table:", len(doctor_stats))
    print(doctor_stats.head(10))

    return doctor_stats


def print_summary(df, doctor_params):
    print("\n--- FINAL SUMMARY ---")
    print("\nExam duration summary:")
    print(df["EXAM_DURATION_MIN"].describe())

    if df["HAS_XRAY"].sum() > 0:
        print("\nX-ray flow summary (for X-ray patients only):")
        print(df.loc[df["HAS_XRAY"] == 1, "XRAY_FLOW_MIN"].describe())

    print("\nPatients with X-ray:", int(df["HAS_XRAY"].sum()))
    print("Patients without X-ray:", int((df["HAS_XRAY"] == 0).sum()))
    print("Unique doctors in cleaned data:", df["DOKTOR_ADI"].nunique())
    print("Doctors in parameter table:", len(doctor_params))


def main():
    print("=== START DATA PROCESSING ===")
    df = load_data()
    df = convert_times(df)
    df = create_features(df)
    df = clean_data(df)
    doctor_params = create_doctor_params(df)
    print_summary(df, doctor_params)

    df.to_excel(OUTPUT_PATH, index=False)
    doctor_params.to_excel(DOCTOR_PARAMS_PATH, index=False)

    print(f"\nSaved cleaned file to: {OUTPUT_PATH}")
    print(f"Saved doctor params to: {DOCTOR_PARAMS_PATH}")


if __name__ == "__main__":
    main()