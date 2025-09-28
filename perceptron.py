import numpy as np

# o - output layer
# h - hidden layer

# функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# производная функции активации
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)
    
# функция потерь
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    def __init__(self):
        # веса и сдвиги (случайные значения)
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    # прямое распространение
    def feed_forward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    # тренировка нейронной сети (градиентный спуск)
    def train(self, data, all_y_trues):
        learn_rate = 0.1 # скорость обучения
        epochs = 1000 # количество эпох обучения

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # feed forward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1) # оно же y_pred

                # считаем частные производные
                d_L_d_ypred = -2 * (y_true - o1) # производная функции потерь (показывает зависимость функции потерь от предсказания)
                
                # нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1) # частная производная предсказания от веса w5 (показывает зависимость предсказания от веса w5)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1) # частная производная предсказания от веса w6 (показывает зависимость предсказания от веса w6)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1) # частная производная предсказания от сдвига b3 (показывает зависимость предсказания от сдвига b3)
                
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1) # частная производная предсказания от h1 (показывает зависимость предсказания от h1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1) # частная производная предсказания от h2 (показывает зависимость предсказания от h2)
                
                # нейрон h1
                d_ypred_d_w1 = x[0] * deriv_sigmoid(sum_h1) # частная производная предсказания от w1 (показывает зависимость предсказания от w1)
                d_ypred_d_w2 = x[1] * deriv_sigmoid(sum_h1) # частная производная предсказания от w2 (показывает зависимость предсказания от w2)
                d_ypred_d_b1 = deriv_sigmoid(sum_h1) # частная производная предсказания от b1 (показывает зависимость предсказания от b1)
                
                # нейрон h2
                d_ypred_d_w3 = x[0] * deriv_sigmoid(sum_h2) # частная производная предсказания от w3 (показывает зависимость предсказания от w3)
                d_ypred_d_w4 = x[1] * deriv_sigmoid(sum_h2) # частная производная предсказания от w4 (показывает зависимость предсказания от w4)
                d_ypred_d_b2 = deriv_sigmoid(sum_h2) # частная производная предсказания от b2 (показывает зависимость предсказания от b2)

                # --- обновление весов и сдвигов ---
                # вычитаем из весов и сдвигов частные производные (то есть влияние соответствующих весов и сдвигов, умноженные на скорость обучения)
                # то есть уменьшаем веса и сдвиги, чтобы уменьшить функцию потерь

                # нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_ypred_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_ypred_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_ypred_d_b1

                # нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_ypred_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_ypred_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_ypred_d_b2

                # нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Считаем потери в конце каждой десятой эпохи обучения ---
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feed_forward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# Массив с нормированными данными (вес, рост)
data = np.array([
    [-2, -1], # Алиса
    [25, 6], # Боб
    [17, 4], # Чарли
    [-15, -6], # Диана
])

# Массив с правильными ответами
all_y_trues = np.array([
    1, # Алиса
    0, # Боб
    0, # Чарли
    1, # Диана
])

network = NeuralNetwork()

# обучаем нейросеть
network.train(data, all_y_trues)

alice = np.array(data[0])
bob = np.array(data[1])
charlie = np.array(data[2])
diana = np.array(data[3])

# новые данные
random_w1 = np.array([-3, -2])
random_w2 = np.array([0, 2])
random_m1 = np.array([12, 8])
random_m2 = np.array([32, 10])

print("Alice: %.3f" % network.feed_forward(alice))
print("Bob: %.3f" % network.feed_forward(bob))
print("Charlie: %.3f" % network.feed_forward(charlie))
print("Diana: %.3f" % network.feed_forward(diana))

print("random_w1: %.3f" % network.feed_forward(random_w1))
print("random_w2: %.3f" % network.feed_forward(random_w2))
print("random_m1: %.3f" % network.feed_forward(random_m1))
print("random_m2: %.3f" % network.feed_forward(random_m2))
