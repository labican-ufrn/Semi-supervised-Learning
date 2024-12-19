from statistics import mean, stdev

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree

# LISTA DE CLASSIFICADORES
list_tree = [
    Tree(),
    Tree(splitter="random"),
    Tree(max_features=None),
    Tree(criterion="entropy"),
    Tree(criterion="entropy", splitter="random"),
    Tree(criterion="entropy", max_features=None),
    Tree(criterion="entropy", max_features=None, splitter="random"),
    Tree(criterion="entropy", max_features='sqrt', splitter="random"),
    Tree(max_features='sqrt', splitter="random"),
    Tree(max_features=None, splitter="random")]

list_knn_Prelax = [
    KNN(n_neighbors=4, weights='distance'),KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5)]

list_knn_seeds = [
    KNN(n_neighbors=4, weights='distance'),KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5),
    KNN(n_neighbors=6, weights='distance'),KNN(n_neighbors=6),
    KNN(n_neighbors=6, weights='distance'),KNN(n_neighbors=6),
    KNN(n_neighbors=6, weights='distance'),KNN(n_neighbors=6)]

list_knn_full = [
    KNN(n_neighbors=4, weights='distance'),KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5),
    KNN(n_neighbors=6, weights='distance'),KNN(n_neighbors=6),
    KNN(n_neighbors=7, weights='distance'),KNN(n_neighbors=7),
    KNN(n_neighbors=8, weights='distance'),KNN(n_neighbors=8)]

def validate_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        msg = "base_estimator ({}) should implement predict_proba!"
        raise ValueError(msg.format(type(estimator).__name__))

def select_labels(y_train, X_train, labelled_percentage):
    """
    Responsável por converter o array de rótulos das instâncias com base
    nas instâncias selecionadas randomicamente.

    Args:
        - y_train (Array): Classes usadas no treinamento
        - X_train (Array): Instâncias
        - labelled_percentage (float): % de instâncias que ficarão com o
            rótulo.

    Returns:
        Retorna o array de classes com base nos rótulos das instância
        selecionadas.
    """
    class_dist = np.bincount(y_train)
    min_acceptable = np.trunc(class_dist * labelled_percentage)
    instances = []

    for lab, cls_dist in enumerate(min_acceptable):
        instances += np.random.choice(
            np.where(y_train == lab)[0],
            int(cls_dist) or 1,
            replace=False
        ).tolist()

    mask = np.ones(len(X_train), bool)
    mask[instances] = 0
    y_train[mask] = -1
    return y_train

def result(option, dataset, y_test, y_pred, path, labelled_level, rounds):
    """
    Responsável por salvar os outputs dos cômites em arquivos
    Args:
        option (int): Opção de escolha do usuário
        dataset (string): Base de dados nome
        y_test (Array): Rótulos usadas para testes
        y_pred (Array): Rótulos predizidos pelo modelo
        comite (): Cômite de classificadores
        labelled_level (float): % que foi selecionada na iteração
    """
    acc = round(accuracy_score(y_test, y_pred) * 100, 4)
    f1 = round(f1_score(y_test, y_pred, average="macro") * 100, 4)
    if option == 1:
        with open(f'{path}/Comite_Naive_.csv', 'a', encoding='utf=8') as f:
            f.write(
                #ROUNDS
                f'\n{rounds},'
                # DATASET
                f'"{dataset}",'
                # LABELLED-LEVEL
                f'{labelled_level},'
                #ACC
                f'{acc},'
                #F1-Score
                f'{f1}'
            )
            f1 = f1_score(y_test, y_pred, average="macro") * 100
        return f1

    if option == 2:
        with open(f'{path}/Comite_Tree_.csv', 'a', encoding='utf=8') as f:
            f.write(
                #ROUNDS
                f'\n{rounds},'
                # DATASET
                f'"{dataset}",'
                # LABELLED-LEVEL
                f'{labelled_level},'
                #ACC
                f'{acc},'
                #F1-Score
                f'{f1}'
            )
            f1 = f1_score(y_test, y_pred, average="macro") * 100
        return f1

    if option == 3:
        with open(f'{path}/Comite_KNN_.csv', 'a', encoding='utf=8') as f:
            f.write(
                #ROUNDS
                f'\n{rounds},'
                # DATASET
                f'"{dataset}",'
                # LABELLED-LEVEL
                f'{labelled_level},'
                #ACC
                f'{acc},'
                #F1-Score
                f'{f1}'
            )
            f1 = f1_score(y_test, y_pred, average="macro") * 100
        return f1

    with open(f'{path}/Comite_Heterogeneo_.csv', 'a', encoding='utf=8') as f:
        f.write(
            #ROUNDS
            f'\n{rounds},'
            # DATASET
            f'"{dataset}",'
            # LABELLED-LEVEL
            f'{labelled_level},'
            #ACC
            f'{acc},'
            #F1-Score
            f'{f1}'
        )
        f1 = f1_score(y_test, y_pred, average="macro") * 100
    return f1

def calculate_mean_stdev(
    fold_result_acc,
    option,
    labelled_level,
    path,
    dataset,
    fold_result_f1_score
):
    """
    Calcula a média de acc e f1_score dependendo do comitê

    Args:
        fold_result_acc (_type_): _description_
        option (_type_): _description_
        labelled_level (_type_): _description_
        path (_type_): _description_
        dataset (_type_): _description_
        fold_result_f1_score (_type_): _description_
    """
    acc_average = round(mean(fold_result_acc), 4)
    standard_deviation_acc = round(stdev(fold_result_acc), 4)
    f1_average = round(mean(fold_result_f1_score), 4)
    standard_deviation_f1 = round(stdev(fold_result_f1_score), 4)
    if option == 1:
        with open(f'{path}/Comite_Naive_F.csv', 'a', encoding='utf=8') as f:
            f.write(
                # DATASET
                f'\n"{dataset}",'
                # LABELLED-LEVEL
                f'{labelled_level},'
                #ACC-AVERAGE
                f'{acc_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_acc},'
                #F1-SCORE-AVERAGE
                f'{f1_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_f1}'
            )
    elif option == 2:
        with open(f'{path}/Comite_Tree_F.csv', 'a') as f:
            f.write(
                # DATASET
                f'\n"{dataset}",'
                # LABELLED-LEVEL
                f'{labelled_level},'
                #ACC-AVERAGE
                f'{acc_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_acc},'
                #F1-SCORE-AVERAGE
                f'{f1_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_f1}'
            )
    elif option == 3:
        with open(f'{path}/Comite_KNN_F.csv', 'a', encoding='utf-8') as f:
            f.write(
                # DATASET
                f'\n"{dataset}",'
                # LABELLED-LEVEL
                f'{labelled_level},'
                #ACC-AVERAGE
                f'{acc_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_acc},'
                #F1-SCORE-AVERAGE
                f'{f1_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_f1}'
            )
    elif option == 4:
        with open(
            f'{path}/Comite_Heterogeneo_F.csv', 'a', encoding='utf-8'
        ) as f:
            f.write(
                # DATASET
                f'\n"{dataset}",'
                # LABELLED-LEVEL
                f'{labelled_level},'
                #ACC-AVERAGE
                f'{acc_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_acc},'
                #F1-SCORE-AVERAGE
                f'{f1_average},'
                #STANDARD-DEVIATION-ACC
                f'{standard_deviation_f1}'
            )
