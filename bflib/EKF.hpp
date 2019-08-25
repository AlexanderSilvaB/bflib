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
        typedef Matrix<dataType, 3, 1> Mat3x1;

    public:
        typedef MatNx1 State;
        typedef MatMx1 Input;
        typedef MatPx1 Output;
        typedef Input Control;
        typedef Output Sensor;
        typedef MatNxN ModelCovariance;
        typedef MatPxP SensorCovariance;
        typedef MatNxN StateMatrix;
        typedef MatNxM InputMatrix;
        typedef MatPxN OutputMatrix;
        typedef MatNxN ModelJacobian;
        typedef MatPxN SensorJacobian;
        typedef MatDx1 Data;
        typedef Mat3x1 Confidence;

    private:
        typedef void (*ModelFunction)(State &x, Input &u, double dt);
        typedef void (*SensorFunction)(Output &y, State &x, Data &d, double dt);
        typedef void (*ModelJacobianFunction)(ModelJacobian &F, State &x, Input &u, double dt);
        typedef void (*SensorJacobianFunction)(SensorJacobian &H, State &x, Data &d, double dt);

        std::default_random_engine gen;
        std::normal_distribution<double> distr{0.0, 1.0};
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        State x, x_1;
        ModelCovariance Q;
        SensorCovariance R;
        Output z;

        ModelJacobian F;
        SensorJacobian H;

        std::vector<Data> dataPoints;
    
        MatNx1 randX;
        MatPx1 randY;

        MatNxN P;
        MatNxN Qsqrt;
        MatPxP Rsqrt;

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

        EKF(State X) : x(X)
        {
            Q.setIdentity();
            R.setIdentity();
            init();
        }

        EKF(ModelCovariance Q, SensorCovariance R) : Q(Q), R(R)
        {
            x.setZero();
            init();
        }

        EKF(State X, ModelCovariance Q, SensorCovariance R) : x(X), Q(Q), R(R)
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

        State state()
        {
            MatNx1 x;
            x.setZero();
            return x;
        }

        Input input()
        {
            MatMx1 u;
            u.setZero();
            return u;
        }

        Output output()
        {
            MatPx1 y;
            y.setZero();
            return y;
        }

        ModelCovariance createQ()
        {
            ModelCovariance Q;
            Q.setZero();
            return Q;
        }

        SensorCovariance createR()
        {
            SensorCovariance R;
            R.setZero();
            return R;
        }

        Data createData()
        {
            Data D;
            D.setZero();
            return D;
        }

        ModelCovariance getP()
        {
            return P;
        }

        Confidence getConfidence(unsigned int x1, unsigned int x2)
        {
            Confidence C;
            C.setZero();
            if(x1 >= states || x2 >= states)
                return C;
            
            Matrix<dataType, 2, 2> p;
            p(0, 0) = P(x1, x1);
            p(0, 1) = P(x1, x2);
            p(1, 0) = P(x2, x1);
            p(1, 1) = P(x2, x2);

            EigenSolver< Matrix<dataType, 2, 2> > es(p);
            Matrix<dataType, 2, 2> eValue = es.pseudoEigenvalueMatrix();
            Matrix<dataType, 2, 2> eVector = es.pseudoEigenvectors();

            C[0] = eValue(0,0);
            C[1] = eValue(1,1);
            C[2] = std::atan2(eVector(0, 1), eVector(0, 0));

            return C;
        }

        void setQ(ModelCovariance Q)
        {
            this->Q = Q;
            P = Q;
            Qsqrt = Q.cwiseSqrt();
        }

        void setR(SensorCovariance R)
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

        void addData(Data &data)
        {
            dataPoints.push_back(data);
        }

        void fillData(std::vector<Data> &data)
        {
            dataPoints = data;
        }

        std::vector<Data>& data()
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

        virtual void model(State &x, Input &u, double dt)
        {

        }

        virtual void sensor(Output &z, State &x, Data &d, double dt)
        {

        }

        virtual void modelJacobian(ModelJacobian &F, State &x, Input &u, double dt)
        {

        }

        virtual void sensorJacobian(SensorJacobian &H, State &x, Data &d, double dt)
        {

        }

        void simulate(State &x, Output &y, Input &u, double dt)
        {
            Data data;
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

        void simulate(State &x, std::vector<Output> &y, Input &u, double dt)
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

        void run(State &xK, Output &y, Input &u, double dt)
        {
            predict(u, dt);

            std::vector<MatDx1> data(1);
            std::vector<MatPx1> ys(1);
            ys[0] = y;
            dataAssoc(ys, data, dt);

            update(y, data[0], dt);
            xK = x;
        }

        void run(State &xK, std::vector<Output> &y, Input &u, double dt)
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
        void predict(Input &u, double dt)
        {
            x_1 = x;
            doModel(x, u, dt);
            doModelJ(F, x_1, u, dt);
            P = F * P * F.transpose() + Q;
        }

        void update(Output &y, Data &d, double dt)
        {
            doSensorJ(H, x, d, dt);

            doSensor(z, x, d, dt);
            S = ( H * P * H.transpose() ) + R;
            K = P * H.transpose() * S.inverse();
            x = x + K * (y - z);
            P = (I - K * H) * P;
        }

        void dataAssoc(std::vector<Output> &y, std::vector<Data> &data, double dt)
        {
            if(dataPoints.size() == 0)
                return;
            MatPx1 v;
            dataType minX, X;
            int minJ;
            Matrix<dataType, 1, 1> Xsq;
            Data d;

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

        void doModel(State &x, Input &u, double dt)
        {
            if(modelFn != NULL)
                modelFn(x, u, dt);
            else
                model(x, u, dt);
        }

        void doSensor(Output &z, State &x, Data &d, double dt)
        {
            if(sensorFn != NULL)
                sensorFn(z, x, d, dt);
            else
                sensor(z, x, d, dt);
        }

        void doModelJ(ModelJacobian &F, State &x, Input &u, double dt)
        {
            if(modelJFn != NULL)
                modelJFn(F, x, u, dt);
            else
                modelJacobian(F, x, u, dt);
        }

        void doSensorJ(SensorJacobian &H, State &x, Data &d, double dt)
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