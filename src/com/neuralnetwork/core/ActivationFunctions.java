package com.neuralnetwork.core;

import com.neuralnetwork.core.interfaces.IActivationFunction;

final public class ActivationFunctions
{

    public static class ThresholdFunction implements IActivationFunction
    {

        @Override
        public double apply(double v)
        {
            return v >= 0f ? 1f : 0f;
        }
    }

    public static class SigmoidFunction implements IActivationFunction.IDifferentiableFunction
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
            return (1.0 / (1.0 + Math.exp(negSlope*v)));
        }

        @Override
        public double derivative(double v)
        {
            double negSlopeV = negSlope * v;
            double exp = Math.exp(negSlopeV);
            double denom = (1.0 + exp);

            return ( (slope * exp) / (denom * denom) );
        }
    }

    public static class SigmoidUnityFunction extends SigmoidFunction
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
            if (v<-15.0) return 0.0;
            else if (v>15.0) return 0.0;
            return super.derivative(v);
        }
    }
}
