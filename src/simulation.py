import simpy
import random
import pandas as pd
import numpy as np

CLEANED_DATA_PATH = "data/cleaned_data.xlsx"

RANDOM_SEED = 42
SIM_TIME = 8 * 60  # 8 hours in minutes

NUM_DOCTORS = 3
NUM_XRAY_ROOMS = 1

WALKIN_INTERARRIVAL_MEAN = 10  # adjustable


def create_schedule_1():
    morning = list(range(0, 4 * 60, 6))        # 08:00-12:00
    afternoon = list(range(5 * 60, 8 * 60, 6)) # 13:00-16:00
    return morning + afternoon


def load_parameters():
    df = pd.read_excel(CLEANED_DATA_PATH)

    exam_mean = df["EXAM_DURATION_MIN"].mean()

    xray_prob = df["HAS_XRAY"].mean()

    # XRAY_FLOW_MIN includes waiting + service in real data.
    # To avoid double-counting queueing in simulation, use a smaller proxy service time.
    # We take a conservative service approximation from the lower quantiles.
    xray_patients = df.loc[df["HAS_XRAY"] == 1, "XRAY_FLOW_MIN"].dropna()

    if len(xray_patients) > 0:
        xray_service_proxy = min(xray_patients.median() * 0.4, 12.0)
        xray_flow_mean = xray_patients.mean()
    else:
        xray_service_proxy = 8.0
        xray_flow_mean = np.nan

    appointment_delay = None
    if "APPOINTMENT_DELAY_MIN" in df.columns:
        valid_appt_delay = df["APPOINTMENT_DELAY_MIN"].dropna()
        if len(valid_appt_delay) > 0:
            appointment_delay = valid_appt_delay

    print("Loaded simulation parameters:")
    print(f"Average exam duration: {exam_mean:.2f} min")
    print(f"Observed X-ray probability: {xray_prob:.4f}")
    print(f"Observed X-ray flow mean: {xray_flow_mean:.2f} min" if not np.isnan(xray_flow_mean) else "Observed X-ray flow mean: N/A")
    print(f"Chosen X-ray service proxy: {xray_service_proxy:.2f} min")

    return {
        "exam_mean": max(exam_mean, 1.0),
        "xray_prob": float(xray_prob),
        "xray_service_mean": max(xray_service_proxy, 1.0),
        "appointment_delay_samples": appointment_delay,
    }


def sample_nonnegative_exponential(mean_value):
    return random.expovariate(1 / max(mean_value, 0.01))


def sample_appointment_delay(delay_series):
    if delay_series is None or len(delay_series) == 0:
        return 0.0
    value = float(random.choice(delay_series.tolist()))
    return max(value, 0.0)


class HospitalSimulation:
    def __init__(self, env, num_doctors, num_xray_rooms, params):
        self.env = env
        self.doctors = simpy.Resource(env, capacity=num_doctors)
        self.xray = simpy.Resource(env, capacity=num_xray_rooms)
        self.params = params

        self.waiting_times_doctor = []
        self.waiting_times_xray = []
        self.total_system_times = []

        self.queue_length_doctor_snapshots = []
        self.queue_length_xray_snapshots = []

        self.completed_patients = 0
        self.completed_xray_patients = 0

    def monitor_queues(self):
        while True:
            self.queue_length_doctor_snapshots.append(len(self.doctors.queue))
            self.queue_length_xray_snapshots.append(len(self.xray.queue))
            yield self.env.timeout(1)

    def patient(self, patient_id, patient_type="walkin"):
        arrival_time = self.env.now

        # Initial doctor visit
        doctor_queue_entry = self.env.now
        with self.doctors.request() as req:
            yield req
            doctor_wait = self.env.now - doctor_queue_entry
            self.waiting_times_doctor.append(doctor_wait)

            exam_time = sample_nonnegative_exponential(self.params["exam_mean"])
            yield self.env.timeout(exam_time)

        # X-ray decision
        if random.random() < self.params["xray_prob"]:
            xray_queue_entry = self.env.now
            with self.xray.request() as req:
                yield req
                xray_wait = self.env.now - xray_queue_entry
                self.waiting_times_xray.append(xray_wait)

                xray_service_time = sample_nonnegative_exponential(self.params["xray_service_mean"])
                yield self.env.timeout(xray_service_time)

            self.completed_xray_patients += 1

            # Secondary screening (same doctor pool; exact same doctor identity not modeled yet)
            doctor_queue_entry_2 = self.env.now
            with self.doctors.request() as req:
                yield req
                doctor_wait_2 = self.env.now - doctor_queue_entry_2
                self.waiting_times_doctor.append(doctor_wait_2)

                second_exam_time = sample_nonnegative_exponential(self.params["exam_mean"] / 2)
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
            # Move to scheduled time
            yield self.env.timeout(max(0, scheduled_time - self.env.now))

            # Arrival randomness around schedule
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

        if self.queue_length_doctor_snapshots:
            print(f"Average doctor queue length: {np.mean(self.queue_length_doctor_snapshots):.2f}")
            print(f"Maximum doctor queue length: {np.max(self.queue_length_doctor_snapshots):.2f}")

        if self.queue_length_xray_snapshots:
            print(f"Average X-ray queue length: {np.mean(self.queue_length_xray_snapshots):.2f}")
            print(f"Maximum X-ray queue length: {np.max(self.queue_length_xray_snapshots):.2f}")


def main():
    random.seed(RANDOM_SEED)

    params = load_parameters()

    env = simpy.Environment()
    hospital = HospitalSimulation(
        env=env,
        num_doctors=NUM_DOCTORS,
        num_xray_rooms=NUM_XRAY_ROOMS,
        params=params,
    )

    appointment_schedule = create_schedule_1()

    env.process(hospital.monitor_queues())
    env.process(hospital.walkin_generator())
    env.process(hospital.appointment_generator(appointment_schedule))

    env.run(until=SIM_TIME)

    hospital.print_results()


if __name__ == "__main__":
    main()