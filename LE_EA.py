import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
import time
from deap import base, creator, tools, algorithms



def extract_languages(langs, levels):

    lang_dicts = {'N':['A','I','B'],'A':['I','B'],'Beginner':['B'],'Intermediate':['I'],'Advanced':['A']}
        
    langs_ordered = []
    
    if len(langs)!=len(levels):
        levels = [levels[0]]*len(langs)
        
    levels = [l if l in lang_dicts.keys() else 'Intermediate' for l in levels]
        
    for i in range(len(langs)):
        langs_ordered+=[langs[i]+level for level in lang_dicts[levels[i]]]
        
    return langs_ordered


def create_solution(sheet, max_people=None):

    people = {}
    languages = set()
    name_dict = {}
    #tables = []

    pool = []
    id_number = 0

    for i, row in sheet.iterrows():
        if type(row[4])==str:
            langs_prac = extract_languages([x.strip() for x in row[4].split(',')],['N']) 
        if type(row[5])==str:
            langs_prac += extract_languages([x.strip() for x in row[5].split(',')],['A'])
        if type(row[7])==str:
            levels = [x.strip() for x in row[7].split(',')]
        else:
            levels=['Intermediate']
        if type(row[7])==str:
            langs_teach = [x.strip() for x in row[6].split(',')]
            langs_teach = extract_languages(langs_teach,levels)

        if not len(langs_prac) or not len(langs_teach):
            continue

        for l_p in langs_prac:
            if l_p in langs_teach:
                langs_teach.remove(l_p)

        if all([langs_teach, langs_prac]):

            people[id_number] = [langs_teach, langs_prac]
            name_dict[id_number] = [row[2],row[3]]
            
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

    return people, languages, table, name_dict

def load_sheet(name):
    sheet = pd.read_excel(name, header=1)
    return sheet




name = "data/sheet_ordered.xlsx"
[people, languages, table, names] = create_solution(load_sheet(name), max_people=None)
print(names)
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
#            print(names[p])
            name = ' '.join([str(i) for i in names[p]])
            
            G.add_node(name)
            node_colors.append("g")

            if solution[p].sum():

                langs, prac, time = np.where(solution[p, :, :, :])

                for i in [0,1] :
                    if prac[i] :
                        G.add_edge(name,"_".join([languages[langs[i]], str(time[i])]))
                    else:
                        G.add_edge("_".join([languages[langs[i]], str(time[i])]),name)

                    edge_colors.append('k')#'['k', 'grey'][i])


    nx.draw_circular(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)


def print_solution(solution):

    solution= solution.copy()
    #ppl = people.keys()
    
    while 1:
        print("cleaning")
        lonelies = ((solution.sum(0)>0).sum(1) == 1).sum(1)>0

        if any(lonelies):
            #ppl = [k if not lonelies[k] for k in ppl]
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

#                print(line)

    draw(solution)


def save_solution(solution):
    
    solution= solution.copy()
    
    while 1:
        print("cleaning")
        lonelies = ((solution.sum(0)>0).sum(1) == 1).sum(1)>0

        if any(lonelies):
            #ppl = [k if not lonelies[k] for k in ppl]
            solution[:,lonelies,:,:]=0
            solution[solution.sum(-1).sum(-1).sum(-1)==1,:,:,:]=0

        else:
            break
        
    file = open('output.txt','w')
        
    for var_time in [0, 1]:
        for lang in range(solution.shape[1]):
            if solution[:,lang,:,var_time].sum():
                
                file.write(languages[lang]+', period '+str(var_time)+':\n')
                
                teachers = ((np.where(solution[:,lang,1,var_time])[0].squeeze()))
                pupils = ((np.where(solution[:,lang,0,var_time])[0].squeeze()))
                
                if len(teachers.shape):
                    names_t = ([' '.join([str(t) for t in names[int(t)]]) for t in teachers])

                else:
                    names_t=([' '.join(names[int(teachers)])])  
                    
                if len(pupils.shape):
                    names_p=([' '.join([str(t) for t in names[int(t)]]) for t in pupils])

                else:
                    names_p=([' '.join(names[int(pupils)])])
                    
                file.write('-teaching:\n')
                [file.write(t+'\n') for t in names_t]
                file.write('-practicing:\n')
                [file.write(p+'\n') for p in names_p]
                file.write('\n')

                
    file.close()
                 
    columns=['first_name','last_name','lang_prac','time_prac','lang_teach','time_teach']
    output = pd.DataFrame(columns=columns)

    for person in range(solution.shape[0]):
        if solution[person].sum():
            lang_t, time_t = np.where(solution[person,:,1,:])
            lang_p, time_p = np.where(solution[person,:,0,:])
            new_row = [str(names[person][0]),str(names[person][1]),languages[lang_p[0]],time_p[0],languages[lang_t[0]],time_t[0]]
            new_row = pd.DataFrame({columns[i]:new_row[i] for i in range(len(columns))},index=[person])
            output = output.append(new_row)
    
    output.to_csv('output.xlsx')    

NGEN = 5000
MU = 2000
LAMBDA = 2000
CXPB = 0.2
MUTPB = 0.8

pop = toolbox.population(n=MU)
hof = tools.HallOfFame(1, similar=np.array_equal)
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
    
save_solution(solution)

