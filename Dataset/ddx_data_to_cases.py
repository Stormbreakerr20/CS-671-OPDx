import json
import pandas as pd
import numpy as np

def read_release_evidences():
    with open('release_evidences.json') as f:
        data = json.load(f)
    return data

def read_release_conditions():
    with open('release_conditions.json') as f:
        data = json.load(f)
    return data

with open('cases.json', 'w') as f:
    f.write('')

conditions = read_release_conditions()
evidences = read_release_evidences()

df = pd.read_csv('test.csv')

cases = {}
c = 0
for index, row in df.iterrows():
    case = ''

    age = row['AGE']
    case = 'Age: ' + str(age) + ', '
    sex = row['SEX']
    if(sex == 'M'):
        case = case + 'Sex: ' + 'Male'
    elif(sex == 'F'):
        case = case + 'Sex:' + ' Female'
    
    pathology = row['PATHOLOGY']
    case = case + '\nGround truth pathology: ' + pathology + '\n'

    init_evidence = row['INITIAL_EVIDENCE']
    case = case + '\nInitial Evidence: ' +  (evidences[init_evidence]['question_en']) + ': Yes\n\n'

    symptoms = []
    antecedents = []


    evs = row['EVIDENCES']
    evs = evs[1:-1]
    evs = str(evs).split(',')
    for i in range(len(evs)):
        evs[i] = evs[i].strip()

    symp = {}
    ant = {}

    for k in evs:
        is_antecedent = False
        
        #check if k contains '_@_V'
        if '_@_V' in k:
            #extract the part of the key before _@_V
            key = k.split('_@_V')[0][1:]
            question = evidences[key]['question_en']
            val_key = 'V' + k.split('_@_V')[1]
            val_key = val_key[:-1]
            answer = evidences[key]["value_meaning"][val_key]['en']
            is_antecedent = evidences[key]['is_antecedent']

        elif '_@_' in k:
            key = k.split('_@_')[0][1:]
            question = 'On a scale of 0 to 10, ' + evidences[key]['question_en']
            answer = k.split('_@_')[1]
            answer = answer[:-1]
            is_antecedent = evidences[key]['is_antecedent']
        
        else:
            key = k[1:-1]
            if key in evidences:
                question = evidences[key]['question_en']
                answer = 'Yes'
                is_antecedent = evidences[key]['is_antecedent']

        if is_antecedent:
            if question not in ant:
                ant[question] = []
                ant[question].append('Yes')
            else:
                ant[question].append('Yes')
        else:
            if question not in symp:
                symp[question] = []
                symp[question].append(answer)
            else:
                symp[question].append(answer)

    case = case + 'Symptoms: \n'
    for key, value in symp.items():
        case = case + key + ': ' + ', '.join(value) + '\n'
    
    case = case + '\nAntecedents: \n'
    for key, value in ant.items():
        case = case + key + ': ' + ', '.join(value) + '\n'

    ddx = row['DIFFERENTIAL_DIAGNOSIS']
    case = case + '\nDifferential Diagnosis: `' + str(ddx) + '`.'

    print("Case:", c)
    print(case)
    print('-----------------------------------')
    c += 1

    if c == 20000:
        break

    cases[c] = case

with open('cases.json', 'w') as f:
    json.dump(cases, f, indent=4)

