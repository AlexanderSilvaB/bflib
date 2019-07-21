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

typedef EKF<float, 1, 1, 1> Sine;

void model(Sine::State &x, Sine::Input &u, double dt)
{
    x << sin(u(0));
}

void sensor(Sine::Output &y, Sine::State &x, Sine::Data &d, double dt)
{
    y << x(0);
}

void modelJ(Sine::ModelJacobian &F, Sine::State &x, Sine::Input &u, double dt)
{
    F << 1;
}

void sensorJ(Sine::SensorJacobian &H, Sine::State &x, Sine::Data &d, double dt)
{
    H << 1;
}

int main(int argc, char *argv[])
{
    float sigma_x = 0.01;
    float sigma_y = 5.0;

    Sine::State x0;
    x0 << 20;

    Sine ekf(x0);
    ekf.setModel(model);
    ekf.setSensor(sensor);
    ekf.setModelJacobian(modelJ);
    ekf.setSensorJacobian(sensorJ);

    Sine::ModelCovariance Q;
    Q << sigma_x*sigma_x;  
    ekf.setQ(Q);

    Sine::SensorCovariance R;
    R << sigma_y*sigma_y;
    ekf.setR(R);


    Sine::State x = x0, xK;

    Sine::Input u;
    Sine::Output y;

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