import simpy
import random
import pandas as pd
import numpy as np

CLEANED_DATA_PATH = "data/cleaned_data.xlsx"
DOCTOR_PARAMS_PATH = "data/doctor_params.xlsx"

RANDOM_SEED = 42
SIM_TIME = 8 * 60  # 8 hours in minutes

NUM_XRAY_ROOMS = 1
WALKIN_INTERARRIVAL_MEAN = 10  # adjustable


def create_schedule_1():
    morning = list(range(0, 4 * 60, 6))        # 08:00-12:00
    afternoon = list(range(5 * 60, 8 * 60, 6)) # 13:00-16:00
    return morning + afternoon


def sample_nonnegative_exponential(mean_value):
    return random.expovariate(1 / max(mean_value, 0.01))


def load_parameters():
    cleaned_df = pd.read_excel(CLEANED_DATA_PATH)
    doctor_df = pd.read_excel(DOCTOR_PARAMS_PATH)

    appointment_delay_samples = None
    if "APPOINTMENT_DELAY_MIN" in cleaned_df.columns:
        valid_delays = cleaned_df["APPOINTMENT_DELAY_MIN"].dropna()
        if len(valid_delays) > 0:
            appointment_delay_samples = valid_delays.tolist()

    doctor_records = []
    for _, row in doctor_df.iterrows():
        doctor_records.append({
            "doctor_name": str(row["DOKTOR_ADI"]),
            "patient_count": int(row["patient_count"]),
            "exam_mean": float(row["exam_mean"]),
            "xray_prob": float(row["xray_prob"]),
            "xray_service_mean": float(row["xray_service_mean"]),
        })

    total_weight = sum(d["patient_count"] for d in doctor_records)

    print("Loaded doctor-based parameters:")
    print(f"Doctors in simulation: {len(doctor_records)}")
    print(f"Total doctor weight: {total_weight}")

    return {
        "appointment_delay_samples": appointment_delay_samples,
        "doctor_records": doctor_records,
    }


def sample_appointment_delay(delay_samples):
    if not delay_samples:
        return 0.0
    value = float(random.choice(delay_samples))
    return max(value, 0.0)


class HospitalSimulation:
    def __init__(self, env, params):
        self.env = env
        self.params = params

        self.xray = simpy.Resource(env, capacity=NUM_XRAY_ROOMS)

        # one resource per doctor
        self.doctor_resources = {
            d["doctor_name"]: simpy.Resource(env, capacity=1)
            for d in params["doctor_records"]
        }

        self.waiting_times_doctor = []
        self.waiting_times_xray = []
        self.total_system_times = []

        self.queue_length_xray_snapshots = []
        self.doctor_queue_snapshots = []

        self.completed_patients = 0
        self.completed_xray_patients = 0

    def choose_doctor(self):
        doctors = self.params["doctor_records"]
        weights = [d["patient_count"] for d in doctors]
        chosen = random.choices(doctors, weights=weights, k=1)[0]
        return chosen

    def monitor_queues(self):
        while True:
            self.queue_length_xray_snapshots.append(len(self.xray.queue))

            total_doctor_queue = sum(len(res.queue) for res in self.doctor_resources.values())
            self.doctor_queue_snapshots.append(total_doctor_queue)

            yield self.env.timeout(1)

    def patient(self, patient_id, patient_type="walkin"):
        arrival_time = self.env.now
        doctor_info = self.choose_doctor()
        doctor_name = doctor_info["doctor_name"]
        doctor_res = self.doctor_resources[doctor_name]

        # first exam with assigned doctor
        doctor_queue_entry = self.env.now
        with doctor_res.request() as req:
            yield req
            doctor_wait = self.env.now - doctor_queue_entry
            self.waiting_times_doctor.append(doctor_wait)

            exam_time = sample_nonnegative_exponential(doctor_info["exam_mean"])
            yield self.env.timeout(exam_time)

        # xray decision depends on same doctor's historical pattern
        if random.random() < doctor_info["xray_prob"]:
            xray_queue_entry = self.env.now
            with self.xray.request() as req:
                yield req
                xray_wait = self.env.now - xray_queue_entry
                self.waiting_times_xray.append(xray_wait)

                xray_service_time = sample_nonnegative_exponential(doctor_info["xray_service_mean"])
                yield self.env.timeout(xray_service_time)

            self.completed_xray_patients += 1

            # secondary screening with SAME doctor
            doctor_queue_entry_2 = self.env.now
            with doctor_res.request() as req:
                yield req
                doctor_wait_2 = self.env.now - doctor_queue_entry_2
                self.waiting_times_doctor.append(doctor_wait_2)

                second_exam_time = sample_nonnegative_exponential(max(doctor_info["exam_mean"] / 2, 1.0))
                yield self.env.timeout(second_exam_time)

        total_system_time = self.env.now - arrival_time
        self.total_system_times.append(total_system_time)
        self.completed_patients += 1

    def walkin_generator(self):
        i = 1
        while True:
            interarrival = sample_nonnegative_exponential(WALKIN_INTERARRIVAL_MEAN)
            yield self.env.timeout(interarrival)
            self.env.process(self.patient(f"W{i}", patient_type="walkin"))
            i += 1

    def appointment_generator(self, schedule_minutes):
        for i, scheduled_time in enumerate(schedule_minutes, start=1):
            yield self.env.timeout(max(0, scheduled_time - self.env.now))
            delay = sample_appointment_delay(self.params["appointment_delay_samples"])
            yield self.env.timeout(delay)
            self.env.process(self.patient(f"A{i}", patient_type="appointment"))

    def print_results(self):
        print("\n=== SIMULATION RESULTS ===")
        print(f"Patients completed: {self.completed_patients}")
        print(f"Patients who used X-ray: {self.completed_xray_patients}")

        if self.waiting_times_doctor:
            print(f"Average doctor wait: {np.mean(self.waiting_times_doctor):.2f} min")
            print(f"Maximum doctor wait: {np.max(self.waiting_times_doctor):.2f} min")

        if self.waiting_times_xray:
            print(f"Average X-ray wait: {np.mean(self.waiting_times_xray):.2f} min")
            print(f"Maximum X-ray wait: {np.max(self.waiting_times_xray):.2f} min")

        if self.total_system_times:
            print(f"Average total system time: {np.mean(self.total_system_times):.2f} min")
            print(f"Maximum total system time: {np.max(self.total_system_times):.2f} min")

        if self.doctor_queue_snapshots:
            print(f"Average total doctor queue length: {np.mean(self.doctor_queue_snapshots):.2f}")
            print(f"Maximum total doctor queue length: {np.max(self.doctor_queue_snapshots):.2f}")

        if self.queue_length_xray_snapshots:
            print(f"Average X-ray queue length: {np.mean(self.queue_length_xray_snapshots):.2f}")
            print(f"Maximum X-ray queue length: {np.max(self.queue_length_xray_snapshots):.2f}")


def main():
    random.seed(RANDOM_SEED)

    params = load_parameters()
    env = simpy.Environment()
    hospital = HospitalSimulation(env, params)

    appointment_schedule = create_schedule_1()

    env.process(hospital.monitor_queues())
    env.process(hospital.walkin_generator())
    env.process(hospital.appointment_generator(appointment_schedule))

    env.run(until=SIM_TIME)

    hospital.print_results()


if __name__ == "__main__":
    main()