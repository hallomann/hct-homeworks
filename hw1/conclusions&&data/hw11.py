import gzip  # модуль gzip для сжатия
import pickle  # модуль pickle для сериализации/сериализации данных
import numpy as np
from joblib import Parallel, delayed  # Parallel, delayed для параллельной обработки
import time
from sklearn.metrics import accuracy_score # accuracy_score для оценки производительности модели
import matplotlib.pyplot as plt

# Загрузка данных
with open("X_train (1)", "rb") as file:
    X_train = pickle.load(file)

with open("Y_train (1)", "rb") as file:
    Y_train = pickle.load(file)

with open("X_test (1)", "rb") as file:
    X_test = pickle.load(file)

with open("Y_test (1)", "rb") as file:
    Y_test = pickle.load(file)

k = 101  # количество ближайших соседей k для учета в kNN

# предварительное вычисление сжатых длин выборок X_train для экономии времени вычисление
start_time = time.time()
Cx2_list = [len(gzip.compress(np.array(x2))) for x2 in X_train]  # использование list comprehension для вычисление сжатия длины всех обучающих выборок
precompute_time = time.time() - start_time
print(f"Precomputing Cx2_list took {precompute_time:.2f} seconds.")

def process_x1(x1):
    """
    Эта функция вычисляет нормализованное расстояние сжатия (NCD) между тестовой выборкой x1 и всеми
    обучающими выборками x2 в X_train. Затем она предсказывает классовую метку для x1 на основе большинства голосов
    среди k-ближайших соседей.
    """
    Cx1 = len(gzip.compress(np.array(x1)))  # вычислить сжатую длину тестового образца x1
    distance_from_x1 = ([])  # инициализируем список для хранения расстояний между x1 и каждым x2

    for x2, Cx2 in zip(X_train, Cx2_list):  # Цикл по каждой обучающей выборке x2 и ее предварительно вычисленной сжатой длине Cx2
        x1x2 = x1 + x2  # конкатенация x1 и x2
        Cx1x2 = len(gzip.compress(np.array(x1x2)))  # вычислить сжатую длину объединенной последовательности x1 x2
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)  # вычислить нормализованное расстояние сжатия (NCD) между x1 и x2
        distance_from_x1.append(ncd)  # добавляем в список

    sorted_idx = np.argsort(np.array(distance_from_x1))  # преобразуем список расстояний в массив, получив индексы, которые позволят отсортировать массив
    top_k_class = np.array(Y_train)[sorted_idx[:k]].tolist()  # получить классы k самых близких соседей
    predict_class = max(set(top_k_class), key=top_k_class.count)  # определим класс, который чаще всего появляется среди k лучших классов
    return predict_class


# Сериальный процессинг
start_time = time.time()
predicts_serial = ([])  # инициализируем список для хранения прогнозов, полученных в результате последовательной обработки
for x1 in (X_test):  # для каждого тестового образца x1 добавим его предсказанный класс к predicts_serial
    predicts_serial.append(process_x1(x1))
serial_time = time.time() - start_time
print(f"Serial processing took {serial_time:.2f} seconds.")


def parallel_processing(n_jobs, backend):
    """
    Эта функция выполняет параллельный процессинг тестовых выборок с использованием Parallel и delayed.
    Она возвращает прогнозы и время, затраченное на обработку.
    Параметры:
    - n_jobs: количество потоков или процессов, выполняемых параллельно.
    - backend: используемая серверная часть для распараллеливания ("loky" для процессов, "threading" для потоков).
    """
    start_time = time.time()
    predicts = Parallel(
        n_jobs=n_jobs, backend=backend
    )(  # joblib.Parallel для распараллеливания функции process_x1 во всех тестовых образцах
        delayed(process_x1)(x1)
        for x1 in X_test  # каждый вызов process_x1 откладывается до выполнения
    )
    parallel_time = time.time() - start_time
    print(
        f"Parallel processing with {n_jobs} jobs and backend '{backend}' took {parallel_time:.2f} seconds."
    )
    return predicts, parallel_time


# измерим производительность при различном n_jobs и backend
num_jobs_list = [1, 2, 4, 8]
backends = ["loky", "threading"]
times = {}  # словарь для хранения времени обработки для каждого сервера


# измерения и построение графика
for backend in backends:
    times[backend] = []
    for n_jobs in num_jobs_list:
        _, parallel_time = parallel_processing(n_jobs, backend)
        times[backend].append(parallel_time)

# отобразим результаты
plt.figure(figsize=(10, 6))
for backend in backends:
    plt.plot(num_jobs_list, times[backend], label=f'Backend: {backend}')

plt.xlabel('Number of Jobs')
plt.ylabel('Time (seconds)')
plt.title('Parallel Processing Time vs Number of Jobs')
plt.legend()
plt.grid(True)
plt.show()
