import pandas as pd
import numpy as np

FILE_PATH = "data/Baltalimani_Data_2025.xlsx"


def load_data():
    df = pd.read_excel(FILE_PATH)
    print("Data shape:", df.shape)
    print("Columns:", df.columns)
    return df


def convert_times(df):
    """
    Convert all time-related columns to datetime format.
    Also prints missing/invalid parsing info.
    """

    time_columns = [
        "CAGRILMA_ZAMANI",
        "TETKIK_ISTEK_SAATI",
        "CEKIM_ZAMANI",
        "MUAYENE_SONLANDIRMA_ZAMANI",
        "MUAYENE_KABUL_ZAMANI",
        "RANDEVU_BASLAMA_SAATI",
        "GIRIS_TARIHI",
    ]

    print("\n--- TIME CONVERSION ---")

    for col in time_columns:
        if col in df.columns:
            print(f"\nProcessing column: {col}")

            before_null = df[col].isnull().sum()

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                dayfirst=True
            )

            after_null = df[col].isnull().sum()

            print(f"Before null: {before_null}")
            print(f"After null: {after_null}")
            print(f"New parse errors: {after_null - before_null}")

        else:
            print(f"{col} not found")

    return df


def create_durations(df):
    print("\n--- CREATING DURATIONS ---")

    # X-ray waiting time
    df["XRAY_WAIT_MIN"] = (
        df["CEKIM_ZAMANI"] - df["TETKIK_ISTEK_SAATI"]
    ).dt.total_seconds() / 60

    # Examination duration
    df["EXAM_DURATION_MIN"] = (
        df["MUAYENE_SONLANDIRMA_ZAMANI"] - df["CAGRILMA_ZAMANI"]
    ).dt.total_seconds() / 60

    print("Duration columns created")

    return df


def clean_data(df):
    print("\n--- CLEANING DATA ---")

    initial_count = len(df)

    # Remove negative values
    df = df[df["XRAY_WAIT_MIN"] >= 0]
    df = df[df["EXAM_DURATION_MIN"] >= 0]

    # Remove extreme outliers
    df = df[df["XRAY_WAIT_MIN"] < 180]
    df = df[df["EXAM_DURATION_MIN"] < 180]

    # Drop missing values
    df = df.dropna()

    final_count = len(df)

    print(f"Initial rows: {initial_count}")
    print(f"Remaining rows: {final_count}")
    print(f"Removed rows: {initial_count - final_count}")

    return df


def main():
    print("=== START DATA PROCESSING ===")

    df = load_data()
    df = convert_times(df)
    df = create_durations(df)
    df = clean_data(df)

    print("\n--- FINAL DATA SUMMARY ---")
    print(df.describe())

    df.to_excel("data/cleaned_data.xlsx", index=False)
    print("\nSaved cleaned_data.xlsx successfully!")


if __name__ == "__main__":
    main()