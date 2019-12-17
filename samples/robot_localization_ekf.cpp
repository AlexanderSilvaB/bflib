#include <bflib/EKF.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

//#undef PLOT_REALTIME

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
    Describes how the states changes according to an input
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
void sensor(Robot::Output &y, Robot::State &x, Robot::Landmark &d, double dt)
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
void sensorJ(Robot::SensorJacobian &H, Robot::State &x, Robot::Landmark &d, double dt)
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
#define DX 250
#define DY 250
void drawGrid(cv::Mat& image, int d, const cv::Scalar& color)
{
    int i = 0;
    cv::Point pi;
    cv::Point pf;

    pi.x = 0;
    pf.x = image.cols;
    while (i < image.rows)
    {
        pi.y = i;
        pf.y = i;
        cv::line(image, pi, pf, color, 1);
        i += d;
    }
    
    i = 0;
    pi.y = 0;
    pf.y = image.rows;
    while (i < image.cols)
    {
        pi.x = i;
        pf.x = i;
        cv::line(image, pi, pf, color, 1);
        i += d;
    }
}

void drawPath(cv::Mat& image, const Robot::State& XR, const vector<double>& X, const vector<double>& Y, const cv::Scalar& color, bool strip)
{
    int S = min(X.size(), Y.size());
    vector<cv::Point> points(S);
    for(int i = 0; i < S; i++)
    {
        points[i] = cv::Point(DX + 20 * X[i], DY + 20 * Y[i]);
    }
    if(strip)
    {
        for(int i = 0; i < S - 1; i += 4)
        {
            cv::line(image, points[i], points[i + 1], color, 1);
        }
    }
    else
        cv::polylines(image, points, false, color, 1);
    cv::circle(image, points.back(), 5, color, CV_FILLED);

    cv::Point pf;
    pf.x = (DX + 20 * XR[0]) + 10 * cos(XR[2]);
    pf.y = (DY + 20 * XR[1]) + 10 * sin(XR[2]);
    cv::line(image, points.back(), pf, color, 2);
}

void drawSensor(cv::Mat& image, const Robot::State& X, const vector< Robot::Output >& Y, const cv::Scalar& color)
{
    cv::Point pt1, ptR;

    ptR.x = DX + 20 * X[0];
    ptR.y = DY + 20 * X[1];

    for(int i = 0; i < Y.size(); i++)
    {
        pt1.x = DX + 20 * ( X[0] + Y[i][0] * cos( Y[i][1] + X[2] ) );
        pt1.y = DY + 20 * ( X[1] + Y[i][0] * sin( Y[i][1] + X[2] ) );
        cv::line(image, ptR, pt1, color, 1);
        cv::circle(image, pt1, 3, color, CV_FILLED);
    }
}

void drawUncertainty(cv::Mat& image, const Robot::State& X, const Robot::Uncertainty& C, const cv::Scalar& color)
{
    double scale = 10000;
    cv::Size size(C[0]*scale, C[1]*scale);
    double angle = C[2] / 3.14 * 180;

    cv::Point center;
    center.x = DX + 20 * X[0];
    center.y = DY + 20 * X[1];

    cv::ellipse(image,
        center, 
        size,
        angle, 
        0, 360, color, 1);
}

void drawLandmarks(cv::Mat& image, const vector<Robot::Landmark>& landmarks, const cv::Scalar& color)
{
    cv::Point pt;

    for(int i = 0; i < landmarks.size(); i++)
    {
        pt.x = DX + 20 * landmarks[i][0];
        pt.y = DY + 20 * landmarks[i][1];
        cv::circle(image, pt, 5, color, CV_FILLED);
    }
}
#endif

int main(int argc, char *argv[])
{
    // Defines the standard deviations for the model and the sensor
    double sigma_x_x = 0.01;
    double sigma_x_y = 0.02;
    double sigma_x_a = 0.03;
    double sigma_y_r = 0.1;
    double sigma_y_b = 0.2;

    // Create a new extended kalman filter for the robot
    Robot::State x0;
    x0 << 0, -5, 0;
    Robot ekf(x0);
    // Sets the system functions
    ekf.setModel(model);
    ekf.setSensor(sensor);
    ekf.setModelJacobian(modelJ);
    ekf.setSensorJacobian(sensorJ);

    // Initialize the system random engine
    long long seed = ekf.seed(1576556569435978183);
    cout << "Seed: " << seed << endl;

    // Sets the model covariance
    Robot::ModelCovariance Q;
    Q << sigma_x_x*sigma_x_x,                    0, sigma_x_x*sigma_x_a,
                            0, sigma_x_y*sigma_x_y, sigma_x_y*sigma_x_a,
                            0,                   0, sigma_x_a*sigma_x_a; 
    ekf.setQ(Q);

    // Sets the sensor covariance
    Robot::SensorCovariance R;
    R << sigma_y_r*sigma_y_r,                   0,
                           0, sigma_y_b*sigma_y_b;
    ekf.setR(R);

    // Simetric map
    bool simetric = false;
    // Create the landmarks
    Robot::Landmark D;
    if(!simetric)
        D << 0, 8;
    else
        D << 8, 8;
    ekf.addData(D);
    if(!simetric)
        D << 4, 5;
    else
        D << 8, -8;
    ekf.addData(D);
    if(!simetric)
        D << 9, 12;
    else
        D << -8, -8;
    ekf.addData(D);
    if(!simetric)
        D << 6, 1;
    else
        D << -8, 8;
    ekf.addData(D);
    if(!simetric)
        D << -2, 2;
    else
        D << 0, 0;
    ekf.addData(D);
    // ----------------

    // Variables to hold the system state, the predicted state, the perfect state and the input
    Robot::State x, xK, xP;
    Robot::Input u;

    x = x0;
    xP = x0;

    // Vector of sensor readings
    vector< Robot::Output > y;
    // We are using 3 readings for the simulation
    y.resize(3); 

    // Initializes the input variable (linear speed = 5.0f m/s ; angular speed = 1.0f rad/s)
    u << 5.0, 1.0;

    // Auxiliary variables to plot
    vector<double> X, Y, XK, YK, XP, YP;

    // Landmarks
    vector<Robot::Landmark> landmarks = ekf.data();

    // Defines the simulation (40s of duration, 0.01s for sample time)
    double T = 40;
    double dt = 0.01;

    // Realtime plot initialization
    #ifdef PLOT_REALTIME
    cv::Mat image(500, 500, CV_8UC3);
    cv::Mat resultImage;
    #endif

    // Run the simulation
    int frame = 0;
    double t = 0;
    while (t < T)
    {
        // Simulates the perfect system
        model(xP, u, dt);
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
        XP.push_back(xP(0));
        YP.push_back(xP(1));

        // Increment the simulation time
        t += dt;

        // Realtime plot
        #ifdef PLOT_REALTIME
        image.setTo(cv::Scalar(255, 255, 255));

        drawGrid(image, 50, cv::Scalar(200, 200, 200));
        drawUncertainty(image, xK, ekf.getUncertainty(0, 1), cv::Scalar(255, 0, 0));
        drawLandmarks(image, landmarks, cv::Scalar(255, 0, 0));
        drawSensor(image, xK, y, cv::Scalar(0, 255, 0));
        drawPath(image, x, X, Y, cv::Scalar(0, 0, 0), false);
        drawPath(image, xK, XK, YK, cv::Scalar(0, 0, 255), true);
        drawPath(image, xP, XP, YP, cv::Scalar(255, 0, 255), false);
        
        flip(image, image, 0);
        //cv::Rect rect(130, 130, 350, 300);
        //resultImage = image(rect);
        resultImage = image;
        cv::imshow("Robot Localization EKF", resultImage);
        int key = cv::waitKey((int)(dt * 1000));
        if(key == 27)
        {
            exit(0);
        }
        else if(key == 'p')
        {
            stringstream ss;
            ss << "robot_localization_ekf_" << (frame++) << ".png";
            cv::imwrite(ss.str(), resultImage);
        }
        else if(key == 'r')
        {
            x(0) = rand() % 10 - 5;
            x(1) = rand() % 10 - 5;
            x(2) = (rand() % 360) * 0.0174533;
            xP = x;
        }
        #endif
    }
    
    // Static Plot
    #ifdef PLOT_REALTIME
    cv::imshow("Robot Localization EKF", image);
    cv::waitKey(0);
    #endif


    #if defined PLOT && ! defined PLOT_REALTIME
    plt::title("Position");
    plt::named_plot("Ideal", XP, YP, "m");
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