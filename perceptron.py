import numpy as np


class Perceptron:
    def __init__(self):
        self.max_iteration = 100
        self.learning_rate = 0.001

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = np.random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = labels[index][0]
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
            print(wx)
            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.learning_rate * (y * x[i])

        return self.w

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


train_features = [[3, 3], [4, 3]]
train_labels = [[1], [1]]
test_features = [[1, 1]]
test_labels = [[-1]]
p = Perceptron()
print(p.train(train_features, train_features))
