
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

// Generate simple data: y = 3x + noise
void generate_data(vector<double>& x, vector<double>& y, int n) {
    srand((unsigned)time(0));
    for (int i = 0; i < n; ++i) {
        double xi = -5 + 10.0 * rand() / RAND_MAX;  // [-5, 5]
        double noise = 0.5 * ((double)rand() / RAND_MAX - 0.5);
        double yi = 3 * xi + noise;
        x.push_back(xi);
        y.push_back(yi);
    }
}

// Compute predictions
vector<double> predict(const vector<double>& x, double w, double b) {
    vector<double> y_pred(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y_pred[i] = w * x[i] + b;
    return y_pred;
}

// Mean Squared Error (MSE) with L2 regularization
double loss_l2(const vector<double>& x, const vector<double>& y, double w, double b, double lambda) {
    double mse = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
        mse += pow((w * x[i] + b - y[i]), 2);
    mse /= x.size();
    double l2 = lambda * w * w;  // L2 penalty (bias not penalized)
    return mse + l2;
}

// Gradient descent training
void train(const vector<double>& x, const vector<double>& y,
    double& w, double& b, double lr, double lambda, int epochs) {
    int n = x.size();
    for (int e = 1; e <= epochs; ++e) {
        double dw = 0.0, db = 0.0;
        for (int i = 0; i < n; ++i) {
            double y_pred = w * x[i] + b;
            double err = y_pred - y[i];
            dw += err * x[i];
            db += err;
        }
        dw = (2.0 / n) * dw + 2.0 * lambda * w; // L2 gradient
        db = (2.0 / n) * db;

        w -= lr * dw;
        b -= lr * db;

        if (e % 50 == 0)
            cout << "Epoch " << e
            << " | Loss: " << loss_l2(x, y, w, b, lambda)
            << " | w=" << w << " | b=" << b << endl;
    }
}

int main() {
    vector<double> x, y;
    generate_data(x, y, 100); // 100 samples

    double w = 0.0, b = 0.0;
    double lr = 0.01;
    double lambda = 1;  // Regularization strength
    int epochs = 500;

    cout << "Training linear regression with L2 regularization..." << endl;
    train(x, y, w, b, lr, lambda, epochs);

    cout << "\nFinal model: y = " << w << "x + " << b << endl;
    cout << "Final loss: " << loss_l2(x, y, w, b, lambda) << endl;

    return 0;
}
