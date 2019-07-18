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

#define g 9.81f
#define l 1.0f
#define m 1.0f
#define b 0.5f

const float PI = 3.14159265358979f;


void model(Matrix<float, 3, 1> &x, Matrix<float, 1, 1> &u, double dt)
{
    Matrix<float, 3, 1> dx;
    dx << l*sin(x(1)), 
          x(2)*dt, 
          -(g/l)*sin(x(1))*dt -(b/(m*l))*x(2)*dt + u(0);
    x(0) = 0;
    x = x + dx;
    
    x(1) = fmod(x(1), PI * 2);
    if(x(1) > PI)
        x(1) -= 2 * PI;
    else if(x(1) < -PI)
        x(1) += 2 * PI;
}

void sensor(Matrix<float, 1, 1> &y, Matrix<float, 3, 1> &x, Matrix<float, -1, 1> &d, double dt)
{
    y << l*sin(x(1));
}

void modelJ(Matrix<float, 3, 3> &F, Matrix<float, 3, 1> &x, Matrix<float, 1, 1> &u, double dt)
{
    F << 1,  0,                     0,
         0,  1,                     dt,
         0,  -(g/l)*cos(x(1))*dt,   -(b/(m*l))*dt;
}

void sensorJ(Matrix<float, 1, 3> &H, Matrix<float, 3, 1> &x, Matrix<float, -1, 1> &d, double dt)
{
    H << 0, l*cos(x(1)), 0;
}

int main(int argc, char *argv[])
{
    float sigma_x_x = 0.0;
    float sigma_x_s = 0.0;
    float sigma_x_v = 0.0;
    float sigma_y_s = 0.2;

    Matrix<float, 3, 1> x0;
    x0 << 0, 0, 0;

    EKF<float, 3, 1, 1> ekf(x0);
    ekf.setModel(model);
    ekf.setSensor(sensor);
    ekf.setModelJacobian(modelJ);
    ekf.setSensorJacobian(sensorJ);
    ekf.seed();

    auto Q = ekf.createQ();
    Q << sigma_x_x*sigma_x_x, sigma_x_x*sigma_x_s, sigma_x_x*sigma_x_v,
         sigma_x_s*sigma_x_x, sigma_x_s*sigma_x_s, sigma_x_s*sigma_x_v,
         sigma_x_v*sigma_x_x, sigma_x_v*sigma_x_s, sigma_x_v*sigma_x_v; 
    ekf.setQ(Q);

    auto R = ekf.createR();
    R << sigma_y_s*sigma_y_s;
    ekf.setR(R);


    auto x = x0;
    auto xK = ekf.state();

    auto u = ekf.input();
    auto y = ekf.output();

    vector<float> X, XK, Y, YK, TS;

    float T = 30;
    float dt = 0.01;
    float t = 0;
    while (t < T)
    {
        if(t < 3)
            u << 0.1;
        else
            u << 0;
        ekf.simulate(x, y, u, dt);
        ekf.run(xK, y, u, dt);

        X.push_back(x(1));
        Y.push_back(y(0));
        XK.push_back(xK(1));
        YK.push_back(xK(0));
        TS.push_back(t);

        t += dt;
    }
    
    #ifdef PLOT
    plt::subplot(2,1,1);
    plt::title("Angular position");
    plt::named_plot("Real", TS, X, "k");
    plt::named_plot("Kalman", TS, XK, "r--");
    plt::legend();
    plt::subplot(2,1,2);
    plt::title("X-Position");
    plt::named_plot("Sensor", TS, Y, "g");
    plt::named_plot("Kalman", TS, YK, "r--");
    plt::legend();
    plt::show();
    #endif


    return 0;
}