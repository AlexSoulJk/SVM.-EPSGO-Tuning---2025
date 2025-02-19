import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import norm

# Создаем синтетический набор данных
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Определяем истинную функцию ошибки (неизвестную для алгоритма)
def true_error(C, sigma):
    svm = SVC(C=10**C, kernel='rbf', gamma=10**sigma)
    return 1 - np.mean(cross_val_score(svm, X, y, cv=5))

# Генерируем сетку значений для визуализации
C_range = np.linspace(-2, 2, 20)
sigma_range = np.linspace(-5, 1, 20)
C_grid, sigma_grid = np.meshgrid(C_range, sigma_range)
error_grid = np.array([[true_error(C, sigma) for C in C_range] for sigma in sigma_range])

# Визуализируем поверхность ошибки
plt.figure(figsize=(8, 6))
plt.contourf(C_grid, sigma_grid, error_grid, levels=20, cmap='coolwarm')
plt.colorbar(label='Ошибка')
plt.xlabel('log C')
plt.ylabel('log Sigma')
plt.title('Поверхность ошибки SVM')
plt.show()

# Реализация Grid Search
def grid_search():
    best_params = None
    best_score = float('inf')
    for C in C_range:
        for sigma in sigma_range:
            score = true_error(C, sigma)
            if score < best_score:
                best_score = score
                best_params = (C, sigma)
    return best_params, best_score

best_grid_params, best_grid_score = grid_search()
print(f"Лучшие параметры (Grid Search): C={10**best_grid_params[0]}, sigma={10**best_grid_params[1]} с ошибкой {best_grid_score:.4f}")

# Реализация GP-based Optimization
sampled_points = np.random.uniform([-2, -5], [2, 1], (10, 2))  # Инициализация случайных точек
sampled_errors = np.array([true_error(p[0], p[1]) for p in sampled_points])

kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e-1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

def expected_improvement(X, gp, y_min):
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide='ignore'):
        Z = (y_min - mu) / sigma
        EI = (y_min - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return EI

def gp_optimization():
    global sampled_points, sampled_errors, gp
    plt.figure(figsize=(10, 6))
    for i in range(20):  # Количество итераций
        gp.fit(sampled_points, sampled_errors)
        candidates = np.random.uniform([-2, -5], [2, 1], (100, 2))
        ei_values = expected_improvement(candidates, gp, min(sampled_errors))
        next_point = candidates[np.argmax(ei_values)]
        sampled_points = np.vstack([sampled_points, next_point])
        sampled_errors = np.append(sampled_errors, true_error(next_point[0], next_point[1]))
        
        # Визуализация процесса оптимизации
        plt.contourf(C_grid, sigma_grid, error_grid, levels=20, cmap='coolwarm')
        plt.colorbar(label='Ошибка')
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='black', label='Изученные точки')
        plt.scatter(next_point[0], next_point[1], c='yellow', edgecolors='black', marker='X', s=100, label='Выбранная точка')
        plt.xlabel('log C')
        plt.ylabel('log Sigma')
        plt.title(f'Итерация {i+1}: Обновление GP-модели')
        plt.legend()
        plt.show()
    return sampled_points[np.argmin(sampled_errors)], min(sampled_errors)

best_gp_params, best_gp_score = gp_optimization()
print(f"Лучшие параметры (GP Optimization): C={10**best_gp_params[0]}, sigma={10**best_gp_params[1]} с ошибкой {best_gp_score:.4f}")
