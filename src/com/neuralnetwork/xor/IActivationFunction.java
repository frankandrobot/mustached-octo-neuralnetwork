package com.neuralnetwork.xor;

public interface IActivationFunction
{
    public double apply(double v);

    public interface IDifferentiableFunction extends IActivationFunction
    {
        public double derivative(double v);
    }

    public class ThresholdFunction implements IActivationFunction
    {

        @Override
        public double apply(double v)
        {
            return v >= 0f ? 1f : 0f;
        }
    }

    public class SigmoidFunction implements IDifferentiableFunction
    {
        private final double slope;
        private final double negSlope;

        public SigmoidFunction(double slope)
        {
            this.slope = slope;
            this.negSlope = -slope;
        }

        @Override
        public double apply(double v)
        {
            return (1f / (1f + Math.exp(negSlope*v)));
        }

        @Override
        public double derivative(double v)
        {
            double negSlopeV = negSlope * v;
            double denom = (1f + Math.exp(negSlopeV));

            return ( (slope * Math.exp(negSlopeV)) / denom * denom );

        }
    }

    public class SigmoidUnityFunction extends SigmoidFunction
    {

        public SigmoidUnityFunction()
        {
            super(1.0);
        }

        @Override
        public double apply(double v)
        {
            if (v < -45.0) return 0.0;
            else if (v > 45.0) return 1.0;
            return super.apply(v);
        }

        @Override
        public double derivative(double v)
        {
            if (v<-15.0) return -1.0;
            else if (v>15.0) return 1.0;
            return super.derivative(v);
        }
    }
}
