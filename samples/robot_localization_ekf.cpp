#include <bflib/EKF.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#if defined PLOT && ! defined PLOT_REALTIME
#include "external/matplotlibcpp.h"
#endif

using namespace std;
#if defined PLOT && ! defined PLOT_REALTIME
namespace plt = matplotlibcpp;
#endif

#ifdef PLOT_REALTIME
#include <opencv2/opencv.hpp>
#endif

/*
    The Kalman filter
    -------------------------
    Is an Extended Kalman Filter
    3 states (x, y and theta)
    2 inputs (linear and angular speeds)
    2 outputs (range and bearing from landmarks)
    2 data variables (x and y of the landmarks)
*/
typedef EKF<double, 3, 2, 2, 2> Robot;

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
}

/*
    The sensor function
    -------------------------
    Describes the sensor output based on the current state and an associated data vector
*/
void sensor(Robot::Output &y, Robot::State &x, Robot::Data &d, double dt)
{
    double dx, dy;
    dx = d(0) - x(0);
    dy = d(1) - x(1);
    y << sqrt( dx * dx + dy * dy ),
         atan2( dy, dx ) - x(2);
}

/*
    The model jacobian 
    -------------------------
    Describes the model jacobian based on the current state and the input
*/
void modelJ(Robot::ModelJacobian &F, Robot::State &x, Robot::Input &u, double dt)
{
    F << 1, 0, -sin( x(2) ) * u(0) * dt,
         0, 1,  cos( x(2) ) * u(0) * dt,
         0, 0,  1;
}

/*
    The sensor jacobian 
    -------------------------
    Describes the sensor jacobian based on the current state and an associated data vector
*/
void sensorJ(Robot::SensorJacobian &H, Robot::State &x, Robot::Data &d, double dt)
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

#ifdef PLOT_REALTIME
void drawPath(cv::Mat& image, const vector<double>& X, const vector<double>& Y, const cv::Scalar& color, bool strip)
{
    int S = min(X.size(), Y.size());
    vector<cv::Point> points(S);
    for(int i = 0; i < S; i++)
    {
        points[i] = cv::Point(250 + 20 * X[i], 100 + 20 * Y[i]);
    }
    if(strip)
    {
        for(int i = 0; i < S - 1; i += 4)
        {
            cv::line(image, points[i], points[i + 1], color, 1, cv::LINE_AA);
        }
    }
    else
        cv::polylines(image, points, false, color, 1, cv::LINE_AA);
    cv::circle(image, points.back(), 5, color, CV_FILLED);
}

void drawSensor(cv::Mat& image, const Robot::State& X, const vector< Robot::Output >& Y, const cv::Scalar& color)
{
    cv::Point pt1, ptR;

    ptR.x = 250 + 20 * X[0];
    ptR.y = 100 + 20 * X[1];

    for(int i = 0; i < Y.size(); i++)
    {
        pt1.x = 250 + 20 * ( X[0] + Y[i][0] * cos( Y[i][1] + X[2] ) );
        pt1.y = 100 + 20 * ( X[1] + Y[i][0] * sin( Y[i][1] + X[2] ) );
        cv::line(image, ptR, pt1, color, 1);
        cv::circle(image, pt1, 3, color, CV_FILLED);
    }
}

void drawConfidence(cv::Mat& image, const Robot::State& X, const Robot::Confidence& C, const cv::Scalar& color)
{
    cv::Size size(C[0]*1000000, C[1]*1000000);
    double angle = C[2] / 3.14 * 180;

    cv::Point center;
    center.x = 250 + 20 * X[0];
    center.y = 100 + 20 * X[1];

    cv::ellipse(image,
        center, 
        size,
        angle, 
        0, 360, color, 1, cv::LINE_4);
}
#endif

int main(int argc, char *argv[])
{
    // Defines the standard deviations for the model and the sensor
    double sigma_x_x = 0.001;
    double sigma_x_y = 0.002;
    double sigma_x_a = 0.003;
    double sigma_y_r = 0.01;
    double sigma_y_b = 0.02;

    // Create a new extended kalman filter for the robot
    Robot ekf;
    // Sets the system functions
    ekf.setModel(model);
    ekf.setSensor(sensor);
    ekf.setModelJacobian(modelJ);
    ekf.setSensorJacobian(sensorJ);

    // Initialize the system random engine
    ekf.seed();

    // Sets the model covariance
    Robot::ModelCovariance Q;
    Q << sigma_x_x*sigma_x_x,                   0,                   0,
                           0, sigma_x_y*sigma_x_y,                   0,
                           0,                   0, sigma_x_a*sigma_x_a; 
    ekf.setQ(Q);

    // Sets the sensor covariance
    Robot::SensorCovariance R;
    R << sigma_y_r*sigma_y_r,                   0,
                           0, sigma_y_b*sigma_y_b;
    ekf.setR(R);

    // Create the landmarks
    Robot::Data D;
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

    // Variables to hold the system state, the predicted state and the input
    Robot::State x, xK;
    Robot::Input u;

    // Vector of sensor readings
    vector< Robot::Output > y;
    // We are using 3 readings for the simulation
    y.resize(3); 

    // Initializes the input variable (linear speed = 1.0f m/s ; angular speed = 0.2f rad/s)
    u << 1.0f, 0.2f;

    // Auxiliary variables to plot
    vector<double> X, Y, XK, YK;

    // Defines the simulation (40s of duration, 0.01s for sample time)
    double T = 40;
    double dt = 0.01;

    // Realtime plot initialization
    #ifdef PLOT_REALTIME
    cv::Mat image(500, 500, CV_8UC3);
    #endif

    // Run the simulation
    double t = 0;
    while (t < T)
    {
        // Simulate one frame to get the sensor readings
        // This is not necessary on a real system as the y vector will come from a real sensor
        ekf.simulate(x, y, u, dt);
        // Run the EKF with the sensor readings
        ekf.run(xK, y, u, dt);

        // Store the system state and the predicted state
        // On a real system the system state isn't available, just the prediction
        X.push_back(x(0));
        Y.push_back(x(1));
        XK.push_back(xK(0));
        YK.push_back(xK(1));

        // Increment the simulation time
        t += dt;

        // Realtime plot
        #ifdef PLOT_REALTIME
        image.setTo(cv::Scalar(255, 255, 255));

        drawConfidence(image, xK, ekf.getConfidence(0, 1), cv::Scalar(255, 0, 0));
        drawPath(image, X, Y, cv::Scalar(0, 0, 0), false);
        drawPath(image, XK, YK, cv::Scalar(0, 0, 255), true);
        drawSensor(image, xK, y, cv::Scalar(0, 255, 0));

        cv::imshow("Robot Localization EKF", image);
        cv::waitKey((int)(dt * 1000));
        #endif
    }
    
    // Static Plot
    #ifdef PLOT_REALTIME
    cv::imshow("Robot Localization EKF", image);
    cv::waitKey(0);
    #endif


    #if defined PLOT && ! defined PLOT_REALTIME
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