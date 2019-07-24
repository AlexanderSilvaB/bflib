#include <bflib/MC.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#ifdef PLOT
#include "external/matplotlibcpp.h"
#endif

using namespace std;
#ifdef PLOT
namespace plt = matplotlibcpp;
#endif

const float PI = 3.14159265358979f;

// Number of sensor readings
const int S = 9;

/*
    The Monte Carlo Filter
    -------------------------
    3 states (x, y and theta)
    2 inputs (linear and angular speeds)
    2 outputs (range and bearing) * S sensors
*/
typedef MC<double, 3, 2, 2, S> Robot;

// Limits of the world
Robot::State minState(0000, 0000, 0);
Robot::State maxState(4680, 3200, 2*PI);

// World lines
vector< Vector4d > lines{   Vector4d(0000,  0000,   4680,   0000),
                            Vector4d(0000,  0000,   0000,   3200),
                            Vector4d(0000,  3200,   4680,   3200),
                            Vector4d(4680,  0000,   4680,   3200),
                            Vector4d(0000,  2280,    920,   2280),
                            Vector4d( 920,  2280,    920,   3200),
                            Vector4d(4190,  2850,   4680,   2850),
                            Vector4d(4190,  2850,   4190,   3200),
                            Vector4d(4030,  0000,   4680,   0650) };

/*
    The model function
    -------------------------
    Describes how the state changes according to an input
*/
void model(Robot::State &x, Robot::Input &u, double dt)
{
    Robot::State dx;
    dx << cos( x(2) ) * u(0) * dt,
          sin( x(2) ) * u(0) * dt, 
          u(1) * dt;
    x = x + dx;

    x(2) = fmod(x(2), PI * 2);
    if(x(2) > PI)
        x(2) -= 2 * PI;
    else if(x(2) < -PI)
        x(2) += 2 * PI;
}

/*
    The sensor function
    -------------------------
    Describes the sensor output based on the current state and an associated data vector
*/
void sensor(vector<Robot::Output> &y, Robot::State &x, double dt)
{
    double eps = 0.1;
    double step = (PI / 2.0) / S;
    double angle;
    int start = (S / 2);
    double x1 = x(0), y1 = x(1), th = x(2);
    double x2, y2, x3, y3, x4, y4, den, px, py, thP, dth, dist, dist_i;

    for(int i = 0, n = -start; i < S; i++, n++)
    {  
        angle = n * step;
        x2 = x1 + cos(th + angle);
        y2 = y1 + sin(th + angle);
        dist = -1;

        for(int j = 0; j < lines.size(); j++)
        {
            x3 = lines[j](0);
            y3 = lines[j](1);
            x4 = lines[j](2);
            y4 = lines[j](3);

            den = ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4));
            if(den != 0)
            {
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/den;
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/den;

                thP = atan2(py - y1, px - x1) - th;
                thP = fmod(thP, PI * 2);
                if(thP > PI)
                    thP -= 2 * PI;

                dth = abs(angle - thP);
                if(dth < eps)
                {
                    if(min(x3, x4)-1 <= px && px <= max(x3, x4)+1 && min(y3, y4)-1 <= py && py <= max(y3, y4)+1)
                    {
                        dist_i = sqrt( (x1 - px)*(x1 - px) + (y1 - py)*(y1 - py) );
                        if(dist < 0)
                            dist = dist_i;
                        else if(dist_i < dist)
                            dist = dist_i;
                    }
                }
            }
        }
        
        y[i] << dist, angle;
    }
}

int main(int argc, char *argv[])
{
    // Defines the standard deviations for the resample and the sensor
    double sigma_x_x = 40;
    double sigma_x_y = 40;
    double sigma_x_a = 0.3;
    double sigma_y_r = 100;
    double sigma_y_b = 0;

    // Number of particles
    int N = 1000 ;

    // Create a new monte carlo filter for the robot with max of 1000 particles
    Robot mc(N, minState, maxState);

    // Sets the system functions
    mc.setModel(model);
    mc.setSensor(sensor);

    // Initialize the system random engine
    mc.seed();

    // Sets the resample std
    Robot::ResampleStd Q;
    Q << sigma_x_x,
         sigma_x_y,
         sigma_x_a;
    mc.setQ(Q);

    // Sets the sensor std
    Robot::SensorStd R;
    R << sigma_y_r,
         sigma_y_b;
    mc.setR(R);

    // Variables to hold the system state, the predicted state and the input
    Robot::State x(2000, 1000, 0);
    Robot::State xP;
    Robot::Input u;

    // Sensor readings
    vector<Robot::Output> y(S);

    // Initializes the input variable (linear speed = 1.0f m/s ; angular speed = 0.2f rad/s)
    u << 1000.0f, 1.57f;
    // u << 0, 0;

    // Auxiliary variables to plot
    vector<double> X, Y, XP, YP;

    // Defines the simulation (3s of duration, 0.01s for sample time)
    double T = 3;
    double dt = 0.01;

    // Run the simulation
    double t = 0;
    while (t < T)
    {
        // Simulate one frame to get the sensor readings
        // This is not necessary on a real system as the y vector will come from a real sensor
        mc.simulate(x, y, u, dt);
        // Run the MC with the sensor readings
        mc.run(xP, y, u, dt);

        // Store the system state and the predicted state
        // On a real system the system state isn't available, just the prediction
        X.push_back(x(0));
        Y.push_back(x(1));
        XP.push_back(xP(0));
        YP.push_back(xP(1));

        cout << x << endl;
        cout << xP << endl;

        // Increment the simulation time
        t += dt;
    }

    #ifdef PLOT
    vector<double> x_, y_;
    plt::title("Position");
    for(int i = 0; i < lines.size(); i++)
    {
        x_.resize(2);
        y_.resize(2);
        x_[0] = lines[i](0);
        y_[0] = lines[i](1);
        x_[1] = lines[i](2);
        y_[1] = lines[i](3);
        plt::plot(x_, y_, "k");
    }
    for(int i = 0; i < mc.particles().size(); i++)
    {
        x_.resize(1);
        y_.resize(1);
        x_[0] = mc.particles()[i](0);
        y_[0] = mc.particles()[i](1);
        plt::plot(x_, y_, "m.");
    }
    plt::named_plot("Real", X, Y, "b");
    x_.resize(1);
    y_.resize(1);
    x_[0] = X[X.size()-1];
    y_[0] = Y[Y.size()-1];
    plt::plot(x_, y_, "bX");
    plt::named_plot("MC", XP, YP, "r");
    x_[0] = XP[XP.size()-1];
    y_[0] = YP[YP.size()-1];
    plt::plot(x_, y_, "rX");
    plt::legend();
    plt::show();
    #endif


    return 0;
}