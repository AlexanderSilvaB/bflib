#include <bflib/MC.hpp>
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
Robot::State minState(0.000, 0.000, 0);
Robot::State maxState(4.680, 3.200, 2*PI);

// World lines
vector< Vector4d > lines{   Vector4d(0.000,  0.000,   4.680,   0.000),
                            Vector4d(0.000,  0.000,   0.000,   3.200),
                            Vector4d(0.000,  3.200,   4.680,   3.200),
                            Vector4d(4.680,  0.000,   4.680,   3.200),
                            Vector4d(0.000,  2.280,   0.920,   2.280),
                            Vector4d(0.920,  2.280,   0.920,   3.200),
                            Vector4d(4.190,  2.850,   4.680,   2.850),
                            Vector4d(4.190,  2.850,   4.190,   3.200),
                            Vector4d(4.030,  0.000,   4.680,   0.650) };

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

#ifdef PLOT_REALTIME
void drawLines(cv::Mat& image, const vector< Vector4d >& lines, const cv::Scalar& color)
{
    for(int i = 0; i < lines.size(); i++)
    {
        cv::line(image, 
            cv::Point(10 + lines[i](0) * 100, 10 + lines[i](1) * 100), 
            cv::Point(10 + lines[i](2) * 100, 10 + lines[i](3) * 100),
            color, 2);
    }
}

void drawParticles(cv::Mat& image, const vector< Robot::State >& PS, const cv::Scalar& color)
{
    for(int i = 0; i < PS.size(); i++)
    {
        cv::circle(image, cv::Point(10 + 100 * PS[i][0], 10 + 100 * PS[i][1]), 2, color, CV_FILLED);
    }
}

void drawPath(cv::Mat& image, const vector<double>& X, const vector<double>& Y, const cv::Scalar& color, bool strip)
{
    int S = min(X.size(), Y.size());
    vector<cv::Point> points(S);
    for(int i = 0; i < S; i++)
    {
        points[i] = cv::Point(10 + 100 * X[i], 10 + 100 * Y[i]);
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

    ptR.x = 10 + 100 * X[0];
    ptR.y = 10 + 100 * X[1];

    for(int i = 0; i < Y.size(); i++)
    {
        pt1.x = 10 + 100 * ( X[0] + Y[i][0] * cos( Y[i][1] + X[2] ) );
        pt1.y = 10 + 100 * ( X[1] + Y[i][0] * sin( Y[i][1] + X[2] ) );
        cv::line(image, ptR, pt1, color, 1);
        cv::circle(image, pt1, 3, color, CV_FILLED);
    }
}
#endif

int main(int argc, char *argv[])
{
    // Defines the standard deviations for the resample and the sensor
    double sigma_x_x = 0.04;
    double sigma_x_y = 0.04;
    double sigma_x_a = 0.30;
    double sigma_y_r = 0.10;
    double sigma_y_b = 0.00;

    // Number of particles
    int N = 500 ;

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
    Robot::State x(2.000, 1.000, 0);
    Robot::State xP;
    Robot::Input u;

    // Sensor readings
    vector<Robot::Output> y(S);

    // Initializes the input variable (linear speed = 1.0f m/s ; angular speed = 0.2f rad/s)
    u << 1.0000f, 1.57f;
    // u << 0, 0;

    // Auxiliary variables to plot
    vector<double> X, Y, XP, YP;

    // Defines the simulation (3s of duration, 0.01s for sample time)
    double T = 5;
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
        mc.simulate(x, y, u, dt);
        // Run the MC with the sensor readings
        mc.run(xP, y, u, dt);

        // Store the system state and the predicted state
        // On a real system the system state isn't available, just the prediction
        X.push_back(x(0));
        Y.push_back(x(1));
        XP.push_back(xP(0));
        YP.push_back(xP(1));

        cout << "x = " << x << endl;
        cout << "xP = " << xP << endl;

        // Increment the simulation time
        t += dt;

        // Realtime plot
        #ifdef PLOT_REALTIME
        image.setTo(cv::Scalar(255, 255, 255));

        drawLines(image, lines, cv::Scalar(0, 0, 0));
        drawParticles(image, mc.particles(), cv::Scalar(255, 0, 0));
        drawPath(image, X, Y, cv::Scalar(0, 0, 0), false);
        drawPath(image, XP, YP, cv::Scalar(0, 0, 255), true);
        drawSensor(image, xP, y, cv::Scalar(0, 255, 0));

        cv::imshow("Robot Localization MC", image);
        cv::waitKey((int)(dt * 1000));
        #endif
    }


    // Static Plot
    #ifdef PLOT_REALTIME
    cv::imshow("Robot Localization MC", image);
    cv::waitKey(0);
    #endif


    #if defined PLOT && ! defined PLOT_REALTIME
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