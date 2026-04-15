import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "data/cleaned_data.xlsx"


def main():
    df = pd.read_excel(FILE_PATH)

    print("Cleaned data shape:", df.shape)

    print("\nXRAY_WAIT_MIN summary:")
    print(df["XRAY_WAIT_MIN"].describe())

    print("\nEXAM_DURATION_MIN summary:")
    print(df["EXAM_DURATION_MIN"].describe())

    # X-ray histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["XRAY_WAIT_MIN"], bins=30)
    plt.title("X-Ray Waiting Time Distribution")
    plt.xlabel("Minutes")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results/xray_wait_hist.png")
    plt.show()

    # Exam histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["EXAM_DURATION_MIN"], bins=30)
    plt.title("Examination Duration Distribution")
    plt.xlabel("Minutes")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results/exam_duration_hist.png")
    plt.show()

    # Doctor analysis
    doctor_summary = df.groupby("DOKTOR_ADI")[["XRAY_WAIT_MIN", "EXAM_DURATION_MIN"]].mean()
    print("\nAverage durations by doctor:")
    print(doctor_summary.sort_values("XRAY_WAIT_MIN", ascending=False).head(10))

    doctor_summary.to_excel("results/doctor_summary.xlsx")


if __name__ == "__main__":
    main()