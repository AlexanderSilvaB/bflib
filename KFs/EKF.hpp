#pragma once

#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <thread>
#include <vector>

using namespace Eigen;

template <typename dataType, int states, int inputs, int outputs, int dataConverter = -1>
class EKF
{
    private:
        typedef Matrix<dataType, states, 1> MatNx1;
        typedef Matrix<dataType, states, states> MatNxN;
        typedef Matrix<dataType, states, inputs> MatNxM;
        typedef Matrix<dataType, states, outputs> MatNxP;
        typedef Matrix<dataType, outputs, states> MatPxN;
        typedef Matrix<dataType, outputs, outputs> MatPxP;
        typedef Matrix<dataType, inputs, 1> MatMx1;
        typedef Matrix<dataType, outputs, 1> MatPx1;
        typedef Matrix<dataType, dataConverter, 1> MatDx1;

        typedef void (*ModelFunction)(MatNx1 &x, MatMx1 &u, double dt);
        typedef void (*SensorFunction)(MatPx1 &y, MatNx1 &x, MatDx1 &d, double dt);
        typedef void (*ModelJacobianFunction)(MatNxN &F, MatNx1 &x, MatMx1 &u, double dt);
        typedef void (*SensorJacobianFunction)(MatPxN &H, MatNx1 &x, MatDx1 &d, double dt);

        std::default_random_engine gen;
        std::normal_distribution<double> distr{0.0, 1.0};
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        MatNx1 x, x_1, randX;
        MatNxN Q, P;
        MatPxP R;
        MatPx1 z, randY;

        std::vector<MatDx1> dataPoints;
    
        MatNxN Qsqrt;
        MatPxP Rsqrt;

        MatNxN F;
        MatPxN H;
        MatPxP S;
        MatNxP K;
        MatNxN I;

        ModelFunction modelFn;
        SensorFunction sensorFn;
        ModelJacobianFunction modelJFn;
        SensorJacobianFunction sensorJFn;

        void init()
        {
            modelFn = NULL;
            sensorFn = NULL;
            modelJFn = NULL;
            sensorJFn = NULL;

            P = Q;
            I.setIdentity();

            Qsqrt = Q.cwiseSqrt();
            Rsqrt = R.cwiseSqrt();

            start = std::chrono::high_resolution_clock::now();
        }
    public:
        EKF()
        {
            Q.setIdentity();
            R.setIdentity();
            x.setZero();
            init();
        }

        EKF(MatNx1 X) : x(X)
        {
            Q.setIdentity();
            R.setIdentity();
            init();
        }

        EKF(MatNxN Q, MatPxP R) : Q(Q), R(R)
        {
            x.setZero();
            init();
        }

        EKF(MatNx1 X, MatNxN Q, MatPx1 R) : x(X), Q(Q), R(R)
        {
            init();
        }

        virtual ~EKF()
        {

        }

        void seed()
        {
            gen = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
        }

        void seed(long long s)
        {
            gen = std::default_random_engine(s);
        }

        MatNx1 state()
        {
            MatNx1 x;
            x.setZero();
            return x;
        }

        MatMx1 input()
        {
            MatMx1 u;
            u.setZero();
            return u;
        }

        MatPx1 output()
        {
            MatPx1 y;
            y.setZero();
            return y;
        }

        MatNxN createQ()
        {
            MatNxN Q;
            Q.setZero();
            return Q;
        }

        MatPxP createR()
        {
            MatPxP R;
            R.setZero();
            return R;
        }

        MatDx1 createData()
        {
            MatDx1 D;
            D.setZero();
            return D;
        }

        void setQ(MatNxN Q)
        {
            this->Q = Q;
            P = Q;
            Qsqrt = Q.cwiseSqrt();
        }

        void setR(MatPxP R)
        {
            this->R = R;
            Rsqrt = R.cwiseSqrt();
        }

        double time()
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            start = std::chrono::high_resolution_clock::now();
            return diff.count();
        }

        double delay(double s)
        {
            double ellapsed = time();
            double remain = s - ellapsed;
            if(remain < 0)
                return ellapsed;
            std::this_thread::sleep_for(std::chrono::nanoseconds((long long)(remain * 1e9)));
            ellapsed += time();
            return ellapsed;
        }

        void addData(MatDx1 &data)
        {
            dataPoints.push_back(data);
        }

        void fillData(std::vector<MatDx1> &data)
        {
            dataPoints = data;
        }

        std::vector<MatDx1>& data()
        {
            return dataPoints;
        }

        void setModel(ModelFunction fn)
        {
            modelFn = fn;
        }

        void setSensor(SensorFunction fn)
        {
            sensorFn = fn;
        }

        void setModelJacobian(ModelJacobianFunction fn)
        {
            modelJFn = fn;
        }

        void setSensorJacobian(SensorJacobianFunction fn)
        {
            sensorJFn = fn;
        }

        virtual void model(MatNx1 &x, MatMx1 &u, double dt)
        {

        }

        virtual void sensor(MatPx1 &z, MatNx1 &x, MatDx1 &d, double dt)
        {

        }

        virtual void modelJacobian(MatNxN &F, MatNx1 &x, MatMx1 &u, double dt)
        {

        }

        virtual void sensorJacobian(MatPxN &H, MatNx1 &x, MatDx1 &d, double dt)
        {

        }

        void simulate(MatNx1 &x, MatPx1 &y, MatMx1 &u, double dt)
        {
            MatDx1 data;
            randn(data);

            if(dataPoints.size() > 0)
                data = dataPoints[0];

            doModel(x, u, dt);
            randn(randX);
            x = x + Qsqrt * randX;

            doSensor(y, x, data, dt);            
            randn(randY);
            y = y + Rsqrt * randY;
        }

        void simulate(MatNx1 &x, std::vector<MatPx1> &y, MatMx1 &u, double dt)
        {
            doModel(x, u, dt);
            randn(randX);
            x = x + Qsqrt * randX;
            
            int j = 0, index = 0;
            if(dataPoints.size() == 0)
                j = -1;

            MatDx1 data;
            for(int i = 0; i < y.size(); i++)
            {
                if(j == -1)
                {
                    randn(data);
                    doSensor(y[i], x, data, dt);
                }
                else
                {
                    doSensor(y[i], x, dataPoints[j], dt);
                    j = rand() % dataPoints.size();
                }
                randn(randY);
                y[i] = y[i] + Rsqrt * randY;
            }
        }

        void run(MatNx1 &xK, MatPx1 &y, MatMx1 &u, double dt)
        {
            predict(u, dt);

            std::vector<MatDx1> data(1);
            std::vector<MatPx1> ys(1);
            ys[0] = y;
            dataAssoc(ys, data, dt);

            update(y, data[0], dt);
            xK = x;
        }

        void run(MatNx1 &xK, std::vector<MatPx1> &y, MatMx1 &u, double dt)
        {
            predict(u, dt);

            if(y.size() > 0)
            {
                std::vector<MatDx1> data(y.size());
                dataAssoc(y, data, dt);

                for(int i = 0; i < y.size(); i++)
                    update(y[i], data[i], dt);
            }
            xK = x;
        }

    private:
        void predict(MatMx1 &u, double dt)
        {
            x_1 = x;
            doModel(x, u, dt);
            doModelJ(F, x_1, u, dt);
            P = F * P * F.transpose() + Q;
        }

        void update(MatPx1 &y, MatDx1 &d, double dt)
        {
            doSensorJ(H, x, d, dt);

            doSensor(z, x, d, dt);
            S = ( H * P * H.transpose() ) + R;
            K = P * H.transpose() * S.inverse();
            x = x + K * (y - z);
            P = (I - K * H) * P;
        }

        void dataAssoc(std::vector<MatPx1> &y, std::vector<MatDx1> &data, double dt)
        {
            if(dataPoints.size() == 0)
                return;
            MatPx1 v;
            dataType minX, X;
            int minJ;
            Matrix<dataType, 1, 1> Xsq;
            MatDx1 d;

            for(int i = 0; i < y.size(); i++)
            {
                minX = 0;
                minJ = -1;
                for (int j = 0; j < dataPoints.size(); j++)
                {
                    d = dataPoints[j];
                    doSensor(z, x, d, dt);
                    v = z - y[i];
                    doSensorJ(H, x, d, dt);
                    S = H * P * H.transpose() + R;
                    Xsq = v.transpose() * S.inverse() * v;
                    X = sqrt(Xsq(0));
                    if(minJ < 0)
                    {
                        minJ = j;
                        minX = X;
                    }
                    else if(X < minX)
                    {
                        minJ = j;
                        minX = X;
                    }
                }
                data[i] = dataPoints[minJ];
            }
        }

        void doModel(MatNx1 &x, MatMx1 &u, double dt)
        {
            if(modelFn != NULL)
                modelFn(x, u, dt);
            else
                model(x, u, dt);
        }

        void doSensor(MatPx1 &z, MatNx1 &x, MatDx1 &d, double dt)
        {
            if(sensorFn != NULL)
                sensorFn(z, x, d, dt);
            else
                sensor(z, x, d, dt);
        }

        void doModelJ(MatNxN &F, MatNx1 &x, MatMx1 &u, double dt)
        {
            if(modelJFn != NULL)
                modelJFn(F, x, u, dt);
            else
                modelJacobian(F, x, u, dt);
        }

        void doSensorJ(MatPxN &H, MatNx1 &x, MatDx1 &d, double dt)
        {
            if(sensorJFn != NULL)
                sensorJFn(H, x, d, dt);
            else
                sensorJacobian(H, x, d, dt);
        }

        template<class T>
        void randn(T &mat)
        {
            for (size_t i = 0; i < mat.rows(); i++)
            {
                for (size_t j = 0; j < mat.cols(); j++)
                {
                    mat(i, j) = distr(gen);
                }
            }
        }

};