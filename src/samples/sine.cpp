#include <KFs/EKF.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#ifdef PLOT
#include "matplotlibcpp.h"
#endif

using namespace std;
#ifdef PLOT
namespace plt = matplotlibcpp;
#endif

void model(Matrix<float, 1, 1> &x, Matrix<float, 1, 1> &u, double dt)
{
    x << sin(u(0));
}

void sensor(Matrix<float, 1, 1> &y, Matrix<float, 1, 1> &x, Matrix<float, -1, 1> &d, double dt)
{
    y << x(0);
}

void modelJ(Matrix<float, 1, 1> &F, Matrix<float, 1, 1> &x, Matrix<float, 1, 1> &u, double dt)
{
    F << 1;
}

void sensorJ(Matrix<float, 1, 1> &H, Matrix<float, 1, 1> &x, Matrix<float, -1, 1> &d, double dt)
{
    H << 1;
}

int main(int argc, char *argv[])
{
    float sigma_x = 0.01;
    float sigma_y = 5.0;

    Matrix<float, 1, 1> x0;
    x0 << 20;

    EKF<float, 1, 1, 1> ekf(x0);
    ekf.setModel(model);
    ekf.setSensor(sensor);
    ekf.setModelJacobian(modelJ);
    ekf.setSensorJacobian(sensorJ);

    auto Q = ekf.createQ();
    Q << sigma_x*sigma_x;  
    ekf.setQ(Q);

    auto R = ekf.createR();
    R << sigma_y*sigma_y;
    ekf.setR(R);


    auto x = ekf.state();
    auto xK = ekf.state();

    auto u = ekf.input();
    auto y = ekf.output();

    vector<float> X, XK, Y, TS;

    float T = 30;
    float dt = 0.1;
    float t = 0;
    while (t < T)
    {
        u << t;
        ekf.simulate(x, y, u, dt);
        ekf.run(xK, y, u, dt);

        X.push_back(x(0));
        Y.push_back(y(0));
        XK.push_back(xK(0));
        TS.push_back(t);

        t += dt;
    }
    
    #ifdef PLOT
    plt::title("Position");
    plt::named_plot("Sensor", TS, Y, "g");
    plt::named_plot("Real", TS, X, "k");
    plt::named_plot("Kalman", TS, XK, "r--");
    plt::legend();
    plt::show();
    #endif


    return 0;
}