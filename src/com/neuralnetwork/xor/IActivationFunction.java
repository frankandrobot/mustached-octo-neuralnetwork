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
            return (double) (1f / (1f + Math.exp(negSlope*v)));
        }

        @Override
        public double derivative(double v)
        {
            double negSlopeV = negSlope * v;
            double denom = (double) (1f + Math.exp(negSlopeV));

            return (double) ( (slope * Math.exp(negSlopeV)) / denom * denom );

        }
    }
}
