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

void model(Matrix<double, 3, 1> &x, Matrix<double, 2, 1> &u, double dt)
{
    Matrix<double, 3, 1> dx;
    dx << cos( x(2) ) * u(0) * dt,
          sin( x(2) ) * u(0) * dt, 
          u(1) * dt;
    x = x + dx;
}

void sensor(Matrix<double, 2, 1> &y, Matrix<double, 3, 1> &x, Matrix<double, 2, 1> &d, double dt)
{
    double dx, dy;
    dx = d(0) - x(0);
    dy = d(1) - x(1);
    y << sqrt( dx * dx + dy * dy ),
         atan2( dy, dx ) - x(2);
}

void modelJ(Matrix<double, 3, 3> &F, Matrix<double, 3, 1> &x, Matrix<double, 2, 1> &u, double dt)
{
    F << 1, 0, -sin( x(2) ) * u(0) * dt,
         0, 1,  cos( x(2) ) * u(0) * dt,
         0, 0,  1;
}

void sensorJ(Matrix<double, 2, 3> &H, Matrix<double, 3, 1> &x, Matrix<double, 2, 1> &d, double dt)
{
    double dx, dy, ds, dv, dn1, dn2;
    dx = d(0) - x(0);
    dy = d(1) - x(1);
    ds = sqrt( dx * dx + dy * dy );
    dv = dy / dx;
    dn1 = 1 + ( dv * dv ) * ( dx * dx );
    dn2 = 1 + ( dv * dv ) * dx;

    H << -dx / ds , -dy / ds ,  0,
        dy / dn1,  -1 / dn2, -1;
}

int main(int argc, char *argv[])
{
    double sigma_x_x = 0.001;
    double sigma_x_y = 0.002;
    double sigma_x_a = 0.003;
    double sigma_y_r = 0.01;
    double sigma_y_b = 0.02;

    EKF<double, 3, 2, 2, 2> ekf;
    ekf.setModel(model);
    ekf.setSensor(sensor);
    ekf.setModelJacobian(modelJ);
    ekf.setSensorJacobian(sensorJ);
    ekf.seed();

    auto Q = ekf.createQ();
    Q << sigma_x_x*sigma_x_x,                   0,                   0,
                           0, sigma_x_y*sigma_x_y,                   0,
                           0,                   0, sigma_x_a*sigma_x_a; 
    ekf.setQ(Q);

    auto R = ekf.createR();
    R << sigma_y_r*sigma_y_r,                   0,
                           0, sigma_y_b*sigma_y_b;
    ekf.setR(R);

    // Landmarks-------
    auto D = ekf.createData();
    D << 0, 8;
    ekf.addData(D);
    D << 4, 5;
    ekf.addData(D);
    D << 9, 12;
    ekf.addData(D);
    D << 6, 1;
    ekf.addData(D);
    D << -2, 2;
    ekf.addData(D);
    // ----------------

    auto x = ekf.state();
    auto xK = ekf.state();

    auto u = ekf.input();

    vector< Matrix<double, 2, 1> > y;
    y.resize(3);

    u << 1.0f, 0.2f;

    vector<double> X, Y, XK, YK;

    double T = 40;
    double dt = 0.01;
    double t = 0;
    while (t < T)
    {
        ekf.simulate(x, y, u, dt);
        ekf.run(xK, y, u, dt);

        X.push_back(x(0));
        Y.push_back(x(1));
        XK.push_back(xK(0));
        YK.push_back(xK(1));

        t += dt;
    }
    
    #ifdef PLOT
    plt::title("Position");
    plt::named_plot("Real", X, Y, "k");
    plt::named_plot("Kalman", XK, YK, "r--");
    auto data = ekf.data();
    vector<double> dx(data.size()), dy(data.size());
    for(int i = 0; i < data.size(); i++)
    {   
        dx[i] = data[i](0);
        dy[i] = data[i](1);
    }
    plt::named_plot("Landmarks", dx, dy, "xb");
    plt::legend();
    plt::show();
    #endif


    return 0;
}