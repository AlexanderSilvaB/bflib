# bflib
Bayesian Filters Library

[![Build Status](https://travis-ci.com/AlexanderSilvaB/bflib.svg?branch=master)](https://travis-ci.com/AlexanderSilvaB/bflib)

This library allows the fast implementation of linear and non-linear system predictors based on Bayesian Filters

## Examples
| | |
|-|-|
| Aircraft takeoff (Linear Kalman Filter): [linear_aircraft.cpp](samples/linear_aircraft.cpp) | Aircraft takeoff (Extended Kalman Filter): [non_linear_aircraft.cpp](samples/non_linear_aircraft.cpp) |
| ![Aircraft takeoff linear example](docs/images/linear_aircraft.png?raw=true "Aircraft takeoff") | ![Aircraft takeoff non-linear example](docs/images/non_linear_aircraft.png?raw=true "Aircraft takeoff") |
| Pendulum (Extended Kalman Filter): [pendulum.cpp](samples/pendulum.cpp) | Sine wave prediction (Extended Kalman Filter): [sine.cpp](samples/sine.cpp) |
| ![Pendulum](docs/images/pendulum.png?raw=true "Pendulum") | ![Sine](docs/images/sine.png?raw=true "Sine") |
| Robot localization (Extended Kalman Filter): [robot_localization_ekf.cpp](samples/robot_localization_ekf.cpp) | Robot localization (Particles Filter): [robot_localization_pf.cpp](samples/robot_localization_pf.cpp) |
| ![Robot Localization Kalman](docs/images/robot_localization_ekf.png?raw=true "Robot Localization Kalman") | ![Robot Localization Monte Carlo](docs/images/robot_localization_pf.png?raw=true "Robot Localization Monte Carlo") |

## Features
* Linear Kalman Filter
* Extended Kalman Filter
* Particles Filter ( Monte Carlo )
* Simulate process and sensors
* Time features
* Access to state uncertainty
* Built-in data association
* Easy integration ( Header-Only )
* Optimized for speed
* Eigen as the only dependecy for the library (Samples may use LibPython and OpenCV for data visualization )


## To-do
- [ ] Unscented Kalman Filter
- [ ] Better documentation

## Example code
This example shows how to predict an aircraft altitude and speed during a takeoff using an approximated linear system.  This process can be described by the following set of equations:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;p[k&plus;1]&space;&=&space;p[k]&space;&plus;&space;v[k]&space;\Delta_t&space;&plus;&space;a&space;\frac{\Delta_t^2}{2}\\&space;v[k&plus;1]&space;&=&space;v[k]&space;&plus;&space;a&space;\Delta_t\\&space;y[k]&space;&=&space;p[k]&space;\end{align*}">
</p>
These equations can be rewritten as a system of state-space linear equations:
<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;x[k&plus;1]&space;&=&space;\begin{vmatrix}&space;1&space;&&space;\Delta_t\\&space;0&space;&&space;1&space;\end{vmatrix}x[k]&space;&plus;&space;\begin{vmatrix}&space;\frac{\Delta_t^2}{2}\\&space;\Delta_t&space;\end{vmatrix}u[k]\\&space;y[k]&space;&=&space;\begin{vmatrix}&space;1&space;&&space;0&space;\end{vmatrix}x[k]&space;\end{align*}">
</p>


```cpp
#include <bflib/KF.hpp>
#include <iostream>

using namespace std;

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

        // Prints the predicted state
        cout << xK << endl;

        // Increments the simulation time
        t += dt;
    }

    return 0;
}
```