import random

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

def plot_gantt_chart(machine_schedules):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20.colors  # Color palette

    yticks = []
    yticklabels = []

    for idx, (machine, schedules) in enumerate(sorted(machine_schedules.items())):
        yticks.append(10 * idx)
        yticklabels.append(f"Machine {machine}")
        for start, end, job_id in schedules:
            ax.broken_barh([(start, end - start)], (10 * idx, 9), facecolors=colors[job_id % len(colors)])

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart of Jobs Scheduling')
    plt.tight_layout()

    plt.show()



class Operation:
    def __init__(self, machine, time):
        self.machine = machine
        self.time = int(time)


class Job:
    def __init__(self, operations):
        self.operations = operations


def input_jobs():
    num_jobs = int(input("Enter the number of jobs: "))
    jobs = []

    for j in range(1, num_jobs + 1):
        print(f"\nEntering details for Job {j}:")
        operations = []
        while True:
            machine = input("Enter machine ID (e.g., M1) or 'done' to finish: ")
            if machine.lower() == 'done':
                break
            time = input(f"Enter processing time for {machine}: ")
            operations.append(Operation(machine, time))
        jobs.append(Job(operations))

    return jobs


def compile_machine_schedule(jobs):
    machine_schedule = {}

    # Gather all machines from all jobs to initialize the dictionary
    for job in jobs:
        for operation in job.operations:
            if operation.machine not in machine_schedule:
                machine_schedule[operation.machine] = [0] * len(jobs)

    # Fill the dictionary with processing times
    for i, job in enumerate(jobs):
        for operation in job.operations:
            machine_schedule[operation.machine][i] = operation.time

    return machine_schedule

#**********************************************
def initialize_population(jobs, size):
    population = []
    for _ in range(size):
        individual = []
        for job_id, job in enumerate(jobs):
            start_time = 0
            for operation in job.operations:
                # Correct job_id usage for zero-based list operations
                individual.append((job_id + 1, operation.machine, start_time, operation.time))  # job_id + 1 to match one-based ID
                start_time += operation.time
        random.shuffle(individual)
        population.append(individual)
    return population




def calculate_fitness(individual, jobs):
    machine_end_times = {}
    job_latest_end_times = [0] * len(jobs)
    machine_schedules = {}

    for job_id, machine, start, duration in individual:
        if machine not in machine_schedules:
            machine_schedules[machine] = []
        # Adjust job_id for zero-based index access
        adjusted_job_id = job_id - 1
        start_time = max(machine_end_times.get(machine, 0), job_latest_end_times[adjusted_job_id])
        end_time = start_time + duration
        machine_end_times[machine] = end_time
        job_latest_end_times[adjusted_job_id] = end_time
        machine_schedules[machine].append((start_time, end_time, job_id))

    makespan = max(machine_end_times.values())
    return makespan, machine_schedules



def selection(population, fitnesses):
    tournament_size = 5
    selected = []
    for _ in range(2):  # Select two parents
        contenders = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.append(min(contenders, key=lambda x: x[1])[0])
    return selected


def crossover(parent1, parent2):
    # Find a suitable crossover point that does not violate job sequences
    job_boundaries = [i + 1 for i in range(1, len(parent1) - 1) if parent1[i][0] != parent1[i + 1][0]]
    if not job_boundaries:
        return [parent1.copy(), parent2.copy()]
    point = random.choice(job_boundaries)
    child1 = parent1[:point] + [op for op in parent2 if op not in parent1[:point]]
    child2 = parent2[:point] + [op for op in parent1 if op not in parent2[:point]]
    return [child1, child2]



def mutation(individual, mutation_rate=0.02):
    if random.random() < mutation_rate:
        job_indices = [i for i, op in enumerate(individual) if individual.count(op) > 1]  # Find indices within the same job
        if len(job_indices) > 1:
            idx1, idx2 = random.sample(job_indices, 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual






def print_schedule(machine_schedules):
    print("\nDetailed Scheduling Info:")
    if isinstance(machine_schedules, dict):
        for machine, schedules in sorted(machine_schedules.items()):
            print(f"Machine {machine}:")
            for start, end, job_id in sorted(schedules, key=lambda x: x[0]):
                print(f"  Job {job_id}: Start at {start}, End at {end}")
    else:
        print("Invalid schedule format.")





def print_gantt_chart(machine_schedules):
    max_time = max(end for schedules in machine_schedules.values() for _, end, _ in schedules)
    chart = [[" " for _ in range(max_time + 1)] for _ in range(len(machine_schedules))]

    for machine_idx, (machine, schedules) in enumerate(sorted(machine_schedules.items())):
        for start, end, job_id in schedules:
            for t in range(start, end):
                chart[machine_idx][t] = str(job_id)

    print("\nGantt Chart:")
    for machine_idx, line in enumerate(chart):
        print(f"Machine {machine_idx + 1}: {''.join(line)}")



def genetic_algorithm(jobs, num_generations, population_size):
    population = initialize_population(jobs, population_size)
    best_fitness = float('inf')
    best_solution = None
    best_schedule = None  # This should be a dictionary

    for generation in range(num_generations):
        new_population = []
        fitnesses = []

        for chromosome in population:
            fitness, machine_schedules = calculate_fitness(chromosome, jobs)
            fitnesses.append(fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = chromosome
                best_schedule = machine_schedules  # Ensure this is correctly updated

        while len(new_population) < population_size:
            parents = selection(population, fitnesses)
            children = crossover(parents[0], parents[1])
            new_population.extend(mutation(child, 0.02) for child in children)

        population = new_population[:population_size]

    return best_solution, best_fitness, best_schedule





def main():
    jobs = input_jobs()
    print("\nJobs and their operations have been entered successfully.")

    for idx, job in enumerate(jobs):
        job_description = "Job_" + str(idx + 1) + ": "
        job_description += " -> ".join(f"{op.machine}[{op.time}]" for op in job.operations)
        print(job_description)

    # Compile the schedule for each machine across all jobs
    machine_schedule = compile_machine_schedule(jobs)

    # Output the compiled machine schedule
    print("\nMachine schedules:")
    for machine, times in machine_schedule.items():
        times_str = ", ".join(str(time) for time in times)
        print(f"{machine}: {times_str}")


# Prepare genetic algorithm parameters
    num_generations = 100
    population_size = 50
    best_solution, best_fitness, best_schedule = genetic_algorithm(jobs, num_generations, population_size)
    print(f"\nBest Fitness (makespan): {best_fitness}")
    print_schedule(best_schedule)  # Pass best_schedule here, not best_solution
    print_gantt_chart(best_schedule)

    plot_gantt_chart(best_schedule)  # Call to plot instead of print

if __name__ == "__main__":
    main()



