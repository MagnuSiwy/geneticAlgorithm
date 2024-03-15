import random as rd
from plotly.subplots import make_subplots   
import plotly.graph_objects as go

# Parameter needed to create the plots
OPTIMAL = 13

INSTANCES = 100
GENERATIONS = 100
MUTATION = 0.1

# Structure of the task in the program:
# (task number, starting point, ending point)



def checkDependencies(queue, taskDependencies):
    penalty = 0
    error = len(queue)
    
    # Iterate over tasks in the queue
    for i in range(0, len(queue)):
        for j in range(0, len(queue)):
            # Skip self-comparison
            if j == i :
                continue
            
            # Check if there is a dependency violation between tasks j and i
            # The dependency is represented by (j+1, i+1) in taskDependencies
            # Also, ensure that the end time of task j is not less than or equal to the start time of task i
            if (queue[j][0] + 1, queue[i][0] + 1) in taskDependencies and not queue[j][-1] <= queue[i][-2]:
                # If there is a violation, penalize by adding 10 to the penalty
                penalty += error

    return penalty


def checkResources(queue, tasksTime, availResources, taskResources):
    penalty = 0
    error = len(queue)
    correctStart = False

    for currTime in range(tasksTime + 1):
        currRes = 0

        # Iterate over each task in the queue
        for eachTask in queue:
            # Check if the current time falls within the execution time of the task
            if eachTask[-2] <= currTime < eachTask[-1]:
                # If so, add the resources required by the task to currRes
                currRes += taskResources[eachTask[0]]
            
            # Check if the task starts at time 0
            if eachTask[-2] == 0:
                correctStart = True
            
            # Check if the current resource usage exceeds the available resources
            if currRes > availResources:
                # If so, penalize by adding 10 times the excess resources to the penalty
                penalty += error * (currRes - availResources)
                break

    # If no task starts at time 0, penalize by adding 15 to the penalty
    if not correctStart:
        penalty += error * 1.5
    
    return penalty


def crossover(parent1, parent2):
    # Select a random crossover point between 0 and the length of the parents
    crossover_point = rd.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point]

    # Iterate over the remaining part of the parents starting from the crossover point
    for i in range(crossover_point, len(parent1)):
        # Create a tuple for the child by combining information from parent1 and parent2
        # (Task ID, Start Time from parent2, End Time calculated based on the difference of start and end times in parent1)
        child.append((parent1[i][0], parent2[i][-2], parent2[i][-2] + (parent1[i][-1] - parent1[i][-2])))
        pass        

    return child


def fitness(queue, availResources, taskDependencies, taskResources):
    # Calculate the maximum time required to complete all tasks in the queue
    tasksTime = max(el[-1] for el in queue)

    # Calculate the penalty by checking both dependencies and resources
    # Check dependency violations using the checkDependencies function
    # Check resource-related violations using the checkResources function
    penalty = checkDependencies(queue, taskDependencies) + checkResources(queue, tasksTime, availResources, taskResources)

    # Calculate the fitness as the inverse of the sum of tasksTime and penalty
    fitness_value = 1 / (tasksTime + penalty)

    return fitness_value



def geneticAlgorithm(seed=0, tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    """
    Returns the best solution found by the advanced genetic algorithm
    :param seed: used to initialize the random number generator
    :param tasks: number of tasks in the task planning problem with resources
    :param resources: number of resources in the task planning problem with resources
    :param task_duration: list of durations of the tasks
    :param task_resource: list of resources required by each task
    :param task_dependencies: list of dependencies (expressed as binary tuples) between tasks
    :return: list with the start time of each task in the best solution found, or empty list if no solution was found
    """

    # Set the seed of random functions
    rd.seed(seed)

    # Initialize solutions list and max_time variable which will be used to limit the random starting time of each task
    solutions = []
    ranking = []
    times = []
    bestSolutionStarts = [None] * tasks
    max_time = sum(task_duration) -  min(task_duration)

    # Create the list of completely random solutions that might not be correct
    for s in range(INSTANCES):
        queue = []

        # The task structure is (task_number, starting time[inclusive], ending time[exclusive])
        for i in range(tasks):
            start_time = rd.randint(0, max_time)
            queue.append( (i, start_time, start_time + task_duration[i]) )
            pass

        # Append randomly starting tasks combined as whole queues to the solutions list
        solutions.append(queue)
        pass

    # Main part of the program that creates the generations
    for i in range(GENERATIONS):
        rankedSolutions = []

        # Rank the solutions and append them to the rankedSolutions list
        for s in solutions:
            s.sort(key=lambda x:x[-2])
            value = fitness(s, resources, task_dependencies, task_resource)
            rankedSolutions.append( (value, s) )
            pass

        # Because the bigger the value of the fitness function the better, sort the rankedSolutions list in a reversed order
        rankedSolutions.sort(reverse=True)

        # For testing purposes, print the best queue, time it takes to end all of the tasks and grading
        print(f"=== Gen {i} best solutions ===")
        timeTaken = max(el[-1] for el in rankedSolutions[0][1]) - min(el[-2] for el in rankedSolutions[0][1])
        print(timeTaken, rankedSolutions[0])

        ranking.append(rankedSolutions[0][0])
        times.append(timeTaken)
        for task in rankedSolutions[0][1]:
            bestSolutionStarts[task[0]] = task[-2]

        # Initialize bestSolutions list that contains the best 10% of all the ranked solutions
        bestSolutions = rankedSolutions[:INSTANCES // 10]
        bestSolutionsProbability = []

        # Set the value of bestSolutionsProbability for each element of bestSolutions - the probability of beeing picked as a parent
        # Remove the ranking grade from the bestSolutions elements
        for i in range(len(bestSolutions)):
            bestSolutionsProbability.append(bestSolutions[i][0])
            bestSolutions[i] = bestSolutions[i][1]
            pass

        newGen = []
        
        # Create the remaining 90% of the new generation as the newGen 
        for _ in range(INSTANCES // 10 * 9):
            # Randomly pick the parents, respecting the probabilities of being picked for each one of them
            parent1 = rd.choices(bestSolutions, weights=bestSolutionsProbability)[0]
            parent2 = rd.choices(bestSolutions, weights=bestSolutionsProbability)[0]
            while parent1 == parent2:
                parent2 = rd.choices(bestSolutions, weights=bestSolutionsProbability)[0]
                pass
            # Do a crossover to create a child
            child = crossover(parent1, parent2)

            # Mutate the children until there are no repetitions
            while True:
                for el in range(len(child)):
                    # Randomly pick the mutation chance
                    mutation_chance = rd.uniform(0, 1)

                    # If it is smaller than MUTATION, mutate the starting time of the task in the child
                    if mutation_chance < MUTATION:
                        start = rd.randint(0, max_time)
                        # Combine the task with the randomly picked starting time and set ending time appropriately
                        child[el] = (child[el][0], start, start + task_duration[child[el][0]])
                        pass
                    pass

                # If the child is unique, then stop the loop
                if not (child in newGen or child in bestSolutions):
                    break
                pass

            # Append the newly created child to the newGen list 
            newGen.append(child)
            pass

        # Combine bestSolutions and the newGen to create new solutions
        solutions = newGen + bestSolutions
        pass


    # Creating the plot for the results
    # divisorOfTheList = 50
    # fig = make_subplots(rows=1, cols=2)
    # fig.add_trace(go.Scatter(
    #     x=list(range(GENERATIONS))[::GENERATIONS // divisorOfTheList],
    #     y=ranking[::GENERATIONS // divisorOfTheList],
    #     mode='lines+markers',
    #     line=dict(color="firebrick", dash="dot"),
    #     marker=dict(size=10)),
    #     row=1, col=1
    # )
    # fig.add_trace(go.Scatter(
    #     x=list(range(0, GENERATIONS))[::GENERATIONS // divisorOfTheList],
    #     y=times[::GENERATIONS // divisorOfTheList],
    #     mode='lines+markers',
    #     line=dict(color="royalblue", dash="dot"),
    #     marker=dict(size=10)),
    #     row=1, col=2
    # )
    # fig.add_hline(y=OPTIMAL, line_dash="dash", row=1, col=2,
    #           annotation_text="Optimal Makespan", 
    #           annotation_position="bottom left", line=dict(color="green"))
    # fig.update_layout(title_text=f"Ranking of the best solution for each generation (Chosen from {INSTANCES} instances)",
    #                 title_font_size=40,
    #                 font_size=13,
    #                 showlegend=False
    # )
                    
    # fig.update_yaxes(title_text="Ranking of the best solution in the generation ( 1 / (time + penalty) )", title_font_size=20, row=1, col=1)
    # fig.update_yaxes(title_text="Time the tasks take in the best queue in generation", title_font_size=23, row=1, col=2)

    # fig.update_xaxes(title_text="Generation", title_font_size=30, range=[0, GENERATIONS], row=1, col=1)
    # fig.update_xaxes(title_text="Generation", title_font_size=30, range=[0, GENERATIONS], row=1, col=2)
    # fig.show()


    return bestSolutionStarts



# Examples

if __name__ == "__main__":
    print(geneticAlgorithm(
        None,
        tasks = 6,
        resources = 4,
        task_duration = [3, 4, 2, 2, 1, 4],
        task_resource = [2, 3, 4, 4, 3, 2],
        task_dependencies = [(1, 3), (2, 3), (2, 4), (3, 5), (4, 6)])),

    # geneticAlgorithm(
    #     None,
    #     tasks = 7,
    #     resources = 5,
    #     task_duration = [2, 1, 1, 1, 3, 2, 1],
    #     task_resource = [4, 1, 2, 2, 2, 1, 2],
    #     task_dependencies = [(1, 3), (1, 5), (3, 6), (4, 6), (5, 7), (6, 7)]),

    # geneticAlgorithm(
    #     None,
    #     tasks = 10,
    #     resources = 6,
    #     task_duration = [3, 2, 5, 4, 2, 3, 4, 2, 4, 6],
    #     task_resource = [5, 1, 1, 1, 3, 3, 2, 4, 5, 2],
    #     task_dependencies = [(1, 4), (1, 5), (2, 9), (2, 10), (3, 8), (4, 6),
    #                         (4, 7), (5, 9), (5, 10), (6, 8), (6, 9), (7, 8)]),

    # geneticAlgorithm(
    #     None,
    #     tasks = 30,
    #     resources = 28,
    #     task_duration = [5, 3, 1, 1, 5, 2, 2, 8, 9, 1,
    #                     4, 2, 4, 1, 1, 3, 7, 6, 1, 8,
    #                     3, 3, 10, 8, 7, 4, 5, 2, 7, 5],
    #     task_resource = [8, 5, 1, 7, 1, 4, 1, 4, 5, 1,
    #                     7, 11, 1, 6, 1, 1, 10, 9, 8, 1,
    #                     1, 1, 1, 8, 8, 1, 1, 9, 1, 3],
    #     task_dependencies = [(1, 4), (2, 5), (2, 16), (2, 25), (3, 9), (3, 13),
    #                         (3, 18), (4, 11), (4, 15), (4, 17), (5, 6), (5, 7),
    #                         (5, 14), (6, 8), (6, 10), (6, 19), (7, 21), (8, 22),
    #                         (9, 20), (10, 12), (11, 16), (11, 30), (12, 28), (13, 26),
    #                         (14, 18), (14, 28), (15, 25), (15, 26), (16, 26), (16,27),
    #                         (17, 18), (17, 24), (18, 27), (19, 24), (20, 29), (21, 23),
    #                         (22, 30), (23, 27), (24, 30), (25, 28), (26, 29), (27, 29)]),