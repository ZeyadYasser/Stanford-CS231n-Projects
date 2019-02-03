#ifndef GATE_H
#define GATE_H


class Gate
{
protected:
    double gradient;
public:
    virtual void Forward() = 0;
    virtual double Gradient() { return gradient; }
    virtual void Backward(double) = 0;
    virtual double Value() = 0;
    virtual ~Gate() {};
};

class AddGate : public Gate
{
private:
    Gate* mX;
    Gate* mY;
    double mValue;
public:
    AddGate(Gate* x, Gate* y) {
        mX = x;
        mY = y;
    }
    void Forward() {
        mValue = mX->Value() + mY->Value();
    }
    void Backward(double dz) {
        mX->Backward(dz);
        mY->Backward(dz);
    }
    double Value() { return mValue; }
    ~AddGate() {}
};

class MulGate : public Gate
{
private:
    Gate* mX;
    Gate* mY;
    double mValue;
public:
    MulGate(Gate* x, Gate* y) {
        mX = x;
        mY = y;
    }
    void Forward() {
        mValue = mX->Value() * mY->Value();
    }
    void Backward(double dz) {
        mX->Backward(dz * mY->Value());
        mY->Backward(dz * mX->Value());
    }
    double Value() { return mValue; }
    ~MulGate() {}
};

class Scalar : public Gate
{
private:
    double mValue;
public:
    Scalar(double x) {
        mValue = x;
    }
    void Forward() {}
    void Backward(double dz) {
        gradient += dz;
    }
    double Value() {
        return mValue;
    }
    ~Scalar() {}
};

Gate* scaler(double x) {
    return (new Scalar(x));
}

Gate* add(Gate* x, Gate* y) {
    return (new AddGate(x, y));
}

Gate* mul(Gate* x, Gate* y) {
    return (new MulGate(x, y));
}


#endif // GATE_H
