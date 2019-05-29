import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
import time
from deap import base, creator, tools, algorithms



def extract_languages(string):
    if not isinstance(string, str):
        return []
    string = re.split('\W+', string)
    string = [s for s in string if s != '']
    return string


def create_solution(sheet, max_people=None):

    people = {}
    languages = set()

    #tables = []

    pool = []
    id_number = 0

    for i, row in sheet.iterrows():
        langs_prac = extract_languages(row[4]) + extract_languages(row[5])
        langs_teach = extract_languages(row[6])

        for l_p in langs_prac:
            if l_p in langs_teach:
                langs_teach.remove(l_p)

        if all([langs_teach, langs_prac]):

            people[id_number] = [langs_teach, langs_prac]

            [languages.add(lang) for lang in langs_teach+langs_prac]
            id_number += 1

            if max_people is not None and i > max_people:
                print('Breaking solution at i = '+str(i))
                break

    languages = list(languages)

    table = np.zeros([len(people.keys()),len(languages),2])
    for p in people.keys():
        for l_teach in people[p][0]:
            table[p,languages.index(l_teach),0]=1
        for l_learn in people[p][1]:
            table[p,languages.index(l_learn),1]=1

    return people, languages, table

def load_sheet(name):
    sheet = pd.read_excel(name, header=2)
    return sheet




name = "data/sheet.xlsx"
[people, languages, table] = create_solution(load_sheet(name), max_people=None)

people_num = {}
for k in people.keys():
    people_num[k] = [[languages.index(lang) for lang in people[k][time]] for time in [0,1]]

solution = np.zeros(table.shape+(2,))

#shape = [people, languages, practice, time]

def eval_solution(solution):

    solution = solution.copy()

    while 1:
        # print("cleaning")
        lonelies = ((solution.sum(0)>0).sum(1) == 1).sum(1)>0

        if any(lonelies):

            solution[:,lonelies,:,:]=0
            solution[solution.sum(-1).sum(-1).sum(-1)==1,:,:,:]=0

        else:
            break

    return(np.int(solution.sum() - (solution.sum(0).std(1)**3).sum()),)


def init_solution(class_individual):

    solution = class_individual(np.zeros(table.shape+(2,)))

    for p in range(len(people.keys())):
        time = np.random.randint(2)
        l_teach = np.random.choice(people_num[p][0])
        l_prac = np.random.choice(people_num[p][1])
        solution[p,l_teach,0,time]=1
        solution[p,l_prac,1,1-time]=1

    return(solution)

def mutate_solution(solution):

    person = np.random.randint(solution.shape[0])

    if np.random.random()>0.2:
        solution[person,:,:,:] = solution[person,:,:,::-1]

    #what is our current practice time?
    time_prac = (np.where(solution[person, :, 1, :].sum(0) > 0)[0][0])

    #
    # change practice language:
    if np.random.random() > 0.5:
        solution[person, :, 1, time_prac] = 0
        lang = np.random.choice(people_num[person][1])
        solution[person, lang, 1, time_prac] = 1

    # change teaching language:
    if np.random.random() > 0.5:
        solution[person, :, 0, 1-time_prac] = 0
        lang = np.random.choice(people_num[person][0])
        solution[person, lang, 0, 1-time_prac] = 1

    return solution,

def cx_solution(solution1, solution2):
    temp = solution1.copy()
    crossover_genes = np.random.random(size=solution1.shape[0])>0.5
    temp[crossover_genes] = solution2[crossover_genes]
    solution2[crossover_genes] = solution1[crossover_genes]
    solution1[crossover_genes] = temp[crossover_genes]

    return(solution1,solution2)


creator.create("Fitness", base.Fitness, weights=[1.0])
creator.create("Individual", np.ndarray, fitness=creator.Fitness)

toolbox = base.Toolbox()
# toolbox.register("init", )
toolbox.register("individual", init_solution, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual )

toolbox.register("evaluate", eval_solution)
toolbox.register("mate", cx_solution)
toolbox.register("mutate", mutate_solution)
toolbox.register("select", tools.selNSGA2)
t_start = time.time()

def draw(solution):
    plt.clf()

    G = nx.DiGraph()

    node_colors = []
    edge_colors = []

    for i in [0,1]:
        for l,lang in enumerate(languages):
            if solution[:,l,:,i].sum():
                G.add_node("_".join([lang, str(i)]))
                node_colors.append("r")


    for p in range(solution.shape[0]):

        if solution[p].sum():

            G.add_node(p)
            node_colors.append("g")

            if solution[p].sum():

                langs, prac, time = np.where(solution[p, :, :, :])

                for i in [0,1] :
                    if prac[i] :
                        G.add_edge(p,"_".join([languages[langs[i]], str(time[i])]))
                    else:
                        G.add_edge("_".join([languages[langs[i]], str(time[i])]),p)

                    edge_colors.append('k')#'['k', 'grey'][i])


    nx.draw_circular(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)


def print_solution(solution):

    solution= solution.copy()

    while 1:
        print("cleaning")
        lonelies = ((solution.sum(0)>0).sum(1) == 1).sum(1)>0

        if any(lonelies):

            solution[:,lonelies,:,:]=0
            solution[solution.sum(-1).sum(-1).sum(-1)==1,:,:,:]=0

        else:
            break
    for time in [0, 1]:
        for lang in range(solution.shape[1]):
            if solution[:,lang,:,time].sum():
                teachers = ','.join([str(p) for p in np.where(solution[:,lang,1,time])])
                pupils = ','.join([str(p) for p in np.where(solution[:,lang,0,time])])
                line = languages[lang]+'_'+str(time)+':'+'|'.join([teachers,pupils])

                print(line)

    draw(solution)

    # plt.imshow(np.concatenate([solution[:, :, 0, 0] + solution[:, :, 1, 0] * 2,
    #                            solution[:, :, 0, 1] + solution[:, :, 1, 1] * 2], axis=1))


NGEN = 1000
MU = 2000
LAMBDA = 100
CXPB = 0.7
MUTPB = 0.2

pop = toolbox.population(n=MU)
hof = tools.HallOfFame(10, similar=np.array_equal)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                          halloffame=hof)

solution = hof[0]

for solution in hof[::-1]:
    print_solution((solution))
    plt.pause(1)

