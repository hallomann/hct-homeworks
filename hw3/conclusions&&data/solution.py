import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Параметры модели
D = 1e-4  # Коэффициент диффузии
a = 0.1
epsilon = 0.01
beta = 0.5
gamma = 1.0
sigma = 0.0

# Параметры сетки
L = 2.5  # Размер области по x и y
N = 256  # Число узлов по каждому направлению
h = L / N  # Шаг сетки
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Параметры временной дискретизации
tau = 0.1  # Шаг по времени
num_steps = 5000  # Число временных шагов

# Инициализация переменных
u = np.zeros((N, N))
v = np.zeros((N, N))

# Задание начальных условий для u(x, y, 0)
u[int(0 * N): int(0.5 * N), int(0 * N): int(0.5 * N)] = 1.0

# Задание начальных условий для v(x, y, 0)
v[int(0 * N): int(0.5 * N), int(0 * N): int(1.0 * N)] = 0.1

# Функция для построения оператора Лапласа с нулевыми граничными условиями Дирихле


def laplacian_operator(N, h):
    main_diag = -4.0 * np.ones(N * N)
    side_diag = np.ones(N * N)
    up_down_diag = np.ones(N * N)

    # Учет граничных узлов
    for i in range(N):
        side_diag[i * N] = 0
        side_diag[i * N - 1] = 0

    diagonals = [
        main_diag,
        side_diag[:-1],
        side_diag[:-1],
        up_down_diag[:-N],
        up_down_diag[:-N],
    ]
    offsets = [0, -1, 1, -N, N]
    laplace = sp.diags(diagonals, offsets, shape=(N * N, N * N), format="csr")

    laplace /= h**2
    return laplace


# Построение матрицы оператора Лапласа
Laplace = laplacian_operator(N, h)

# Предварительный расчет матрицы для неявного шага (A matrix)
I_i = sp.eye(N * N, format="csr")
A = I_i - tau * D * Laplace

# Функция для преобразования матрицы 2D в 1D и обратно


def to_1d(u):
    return u.ravel()


def to_2d(u_flat):
    return u_flat.reshape((N, N))


# Массивы для хранения результатов
u_list_spsolve = []
v_list_spsolve = []
u_list_cg = []
v_list_cg = []

# Таймеры для измерения времени выполнения
time_spsolve = 0
time_cg = 0

# --------------------------------------------
#             Прямой метод (spsolve)
# --------------------------------------------

# Копирование начальных условий
u_spsolve = u.copy()
v_spsolve = v.copy()

print("Начало расчета с использованием spsolve...")
start_time = time.time()

for step in range(num_steps):
    # Решение для v (явный метод)
    v_spsolve += tau * epsilon * (beta * u_spsolve - gamma * v_spsolve - sigma)

    # Формирование правой части для u
    f_u = u_spsolve + tau * (u_spsolve * (1 - u_spsolve)
                             * (u_spsolve - a) - v_spsolve)

    # Применение граничных условий (u = 0 на границах)
    f_u[0, :] = 0
    f_u[-1, :] = 0
    f_u[:, 0] = 0
    f_u[:, -1] = 0

    # Преобразование в 1D
    f_u_flat = to_1d(f_u)

    # Решение системы линейных уравнений
    u_new_flat = spla.spsolve(A, f_u_flat)
    u_spsolve = to_2d(u_new_flat)

    # Применение граничных условий
    u_spsolve[0, :] = 0
    u_spsolve[-1, :] = 0
    u_spsolve[:, 0] = 0
    u_spsolve[:, -1] = 0

    # Сохранение результатов на определенных шагах для визуализации
    if step % 100 == 0:
        u_list_spsolve.append(u_spsolve.copy())
        v_list_spsolve.append(v_spsolve.copy())

    # Отображение прогресса
    if step % 500 == 0:
        print(f"Шаг {step}/{num_steps}")

end_time = time.time()
time_spsolve = end_time - start_time
print(f"Время выполнения с использованием spsolve: {time_spsolve:.2f} секунд")

# --------------------------------------------
#         Итерационный метод (CG Solver)
# --------------------------------------------

# Копирование начальных условий
u_cg = u.copy()
v_cg = v.copy()

# Предварительное вычисление факторизации для предобуславливания
M2 = spla.spilu(A)


def M_x(x):
    return M2.solve(x)


M = spla.LinearOperator((N * N, N * N), M_x)

print("\nНачало расчета с использованием метода сопряженных градиентов...")
start_time = time.time()

for step in range(num_steps):
    # Решение для v (явный метод)
    v_cg += tau * epsilon * (beta * u_cg - gamma * v_cg - sigma)

    # Формирование правой части для u
    f_u = u_cg + tau * (u_cg * (1 - u_cg) * (u_cg - a) - v_cg)

    # Применение граничных условий (u = 0 на границах)
    f_u[0, :] = 0
    f_u[-1, :] = 0
    f_u[:, 0] = 0
    f_u[:, -1] = 0

    # Преобразование в 1D
    f_u_flat = to_1d(f_u)

    # Начальное приближение
    u0_flat = to_1d(u_cg)

    # Решение системы линейных уравнений методом сопряженных градиентов
    u_new_flat, info = spla.cg(A, f_u_flat, x0=u0_flat, M=M, maxiter=1000)
    if info != 0:
        print(f"Метод CG не сошелся на шаге {step}, info = {info}")
        break

    u_cg = to_2d(u_new_flat)

    # Применение граничных условий
    u_cg[0, :] = 0
    u_cg[-1, :] = 0
    u_cg[:, 0] = 0
    u_cg[:, -1] = 0

    # Сохранение результатов на определенных шагах для визуализации
    if step % 100 == 0:
        u_list_cg.append(u_cg.copy())
        v_list_cg.append(v_cg.copy())

    # Отображение прогресса
    if step % 500 == 0:
        print(f"Шаг {step}/{num_steps}")

end_time = time.time()
time_cg = end_time - start_time
print(f"Время выполнения с использованием метода CG: {time_cg:.2f} секунд")

# --------------------------------------------
#                 Результаты
# --------------------------------------------

# Вычисление ускорения
acceleration = time_spsolve / time_cg
print(f"\nУскорение в: {acceleration:.2f} раз(-а)")

# Визуализация результатов


def animate_solution(u_list, title):
    fig = plt.figure(figsize=(6, 6))
    ims = []
    for u in u_list:
        im = plt.imshow(
            u, animated=True, cmap="hot", origin="lower", extent=(0, L, 0, L)
        )
        ims.append([im])

    plt.title(title)
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    plt.show()
    return ani


# Анимация для решения с использованием spsolve
ani_spsolve = animate_solution(
    u_list_spsolve, "Решение с использованием spsolve")
ani_spsolve.save('spsolve_animation.mp4', writer='ffmpeg')

# Анимация для решения с использованием CG
ani_cg = animate_solution(u_list_cg, "Решение с использованием CG")
ani_cg.save('cg_animation.mp4', writer='ffmpeg')

# График зависимости времени выполнения от числа временных шагов
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_steps + 1),
         [time_spsolve / num_steps] * num_steps, label='spsolve')
plt.plot(range(1, num_steps + 1),
         [time_cg / num_steps] * num_steps, label='CG')
plt.xlabel('Число временных шагов')
plt.ylabel('Время выполнения (секунд)')
plt.title('Зависимость времени выполнения от числа временных шагов')
plt.legend()
plt.show()
