from py2neo import Graph, NodeMatcher
import csv

def get_Disease_dict():
    cypher = 'MATCH (n:Disease) RETURN n.name'
    res = graph.run(cypher).data()  #
    data_Disease = []

    for i in res:
        data_Disease.append(i['n.name'])

    return data_Disease

def get_Check_dict():
    cypher = 'MATCH (n:Check) RETURN n.name'
    res = graph.run(cypher).data()  #
    data_Check = []

    for i in res:
        data_Check.append(i['n.name'])

    return data_Check

def get_Department_dict():
    cypher = 'MATCH (n:Department) RETURN n.name'
    res = graph.run(cypher).data()  #
    data_Department = []

    for i in res:
        data_Department.append(i['n.name'])

    return data_Department

def get_Drug_dict():
    cypher = 'MATCH (n:Drug) RETURN n.name'
    res = graph.run(cypher).data()  #
    data_Drug = []

    for i in res:
        data_Drug.append(i['n.name'])

    return data_Drug

def get_Food_dict():
    cypher = 'MATCH (n:Food) RETURN n.name'
    res = graph.run(cypher).data()  #
    data_Food = []

    for i in res:
        data_Food.append(i['n.name'])

    return data_Food

def get_Producer_dict():
    cypher = 'MATCH (n:Producer) RETURN n.name'
    res = graph.run(cypher).data()  #
    data_Producer = []

    for i in res:
        data_Producer.append(i['n.name'])

    return data_Producer

def get_Symptom_dict():
    cypher = 'MATCH (n:Symptom) RETURN n.name'
    res = graph.run(cypher).data()  #
    data_Symptom = []

    for i in res:
        data_Symptom.append(i['n.name'])

    return data_Symptom

def get_dict():
    data_Disease = get_Disease_dict()
    data_Check = get_Check_dict()
    data_Department = get_Department_dict()
    data_Drug = get_Drug_dict()
    data_Food = get_Food_dict()
    data_Producer = get_Producer_dict()
    data_Symptom = get_Symptom_dict()
    return data_Disease,data_Check,data_Department,data_Drug,data_Food,data_Producer,data_Symptom


def export_data():
    data_Disease,data_Check,data_Department,data_Drug,data_Food,data_Producer,data_Symptom = get_dict()
    f_drug = open('.\dict\Query\drug.txt', 'w+')
    f_food = open('.\dict\Query\\food.txt', 'w+')
    f_check = open('.\dict\Query\check.txt', 'w+')
    f_department = open('.\dict\Query\department.txt', 'w+')
    f_producer = open('.\dict\Query\producer.txt', 'w+')
    f_symptom = open('.\dict\Query\symptoms.txt', 'w+')
    f_disease = open('.\dict\Query\disease.txt', 'w+')

    f_drug.write('\n'.join(list(data_Drug)))
    f_food.write('\n'.join(list(data_Food)))
    f_check.write('\n'.join(list(data_Check)))
    f_department.write('\n'.join(list(data_Department)))
    f_producer.write('\n'.join(list(data_Producer)))
    f_symptom.write('\n'.join(list(data_Symptom)))
    f_disease.write('\n'.join(list(data_Disease)))

    f_drug.close()
    f_food.close()
    f_check.close()
    f_department.close()
    f_producer.close()
    f_symptom.close()
    f_disease.close()


if __name__ == '__main__':
    graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "12345678"),name="neo4j")
    export_data()
    print(1)