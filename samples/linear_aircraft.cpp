#include <bflib/KF.hpp>
#include <iostream>
#include <vector>
#ifdef PLOT
#include "external/matplotlibcpp.h"
#endif

using namespace std;
#ifdef PLOT
namespace plt = matplotlibcpp;
#endif

typedef KF<float, 2, 1, 1> Aircraft;

// The process model
void process(Aircraft::StateMatrix &A, Aircraft::InputMatrix &B, Aircraft::OutputMatrix &C, double dt)
{
    // Fills the A, B and C matrixes of the process
    A << 1, dt,
         0, 1;
    
    B << (dt*dt)/2.0,
         dt; 

    C << 1, 0;
}

int main(int argc, char *argv[])
{
    // The system standard deviation
    float sigma_x_s = 0.01; // std for position
    float sigma_x_v = 0.02; // std for speed
    float sigma_y_s = 5.0; // std for position sensor

    // Creates a linear kalman filter with float data type, 2 states, 1 input and 1 output
    Aircraft kf;

    // Sets the process
    kf.setProcess(process);

    // Creates a new process covariance matrix Q
    Aircraft::ModelCovariance Q;
    // Fills the Q matrix
    Q << sigma_x_s*sigma_x_s, sigma_x_s*sigma_x_v,
         sigma_x_v*sigma_x_s, sigma_x_v*sigma_x_v;  
    // Sets the new Q to the KF
    kf.setQ(Q);

    // Creates a new sensor covariance matrix R
    Aircraft::SensorCovariance R;
    // Fills the R matrix
    R << sigma_y_s*sigma_y_s;
    // Sets the new R to the KF
    kf.setR(R);

    // Creates two states vectors, one for the simulation and one for the kalman output
    Aircraft::State x, xK;

    // Creates an input vector and fills it
    Aircraft::Input u;
    u << 0.1;

    // Creates an output vector
    Aircraft::Output y;

    // Creates auxialiary vector just for plotting purposes
    vector<float> X, XK, Y, TS, V, VK;

    // Defines the simulation max time and the sample time
    float T = 30;
    float dt = 0.1;
    // Creates a variable to hold the time 
    float t = 0;
    while (t < T)
    {
        // Simulate the system in order to obtain the sensor reading (y).
        // It is not needed on a real system
        kf.simulate(x, y, u, dt);
        // Run the kalman filter
        kf.run(xK, y, u, dt);

        // Stores the data to plot later
        X.push_back(x(0));
        V.push_back(x(1));
        Y.push_back(y(0));
        XK.push_back(xK(0));
        VK.push_back(xK(1));
        TS.push_back(t);

        // Increments the simulation time
        t += dt;
    }
    

    // Plotting the results
    #ifdef PLOT
    plt::subplot(2, 1, 1);
    plt::title("Position");
    plt::named_plot("Sensor", TS, Y, "g");
    plt::named_plot("Real", TS, X, "k");
    plt::named_plot("Kalman", TS, XK, "r--");
    plt::legend();
    plt::subplot(2, 1, 2);
    plt::title("Speed");
    plt::named_plot("Real", TS, V, "k");
    plt::named_plot("Kalman", TS, VK, "r--");
    plt::legend();
    plt::show();
    #endif


    return 0;
}