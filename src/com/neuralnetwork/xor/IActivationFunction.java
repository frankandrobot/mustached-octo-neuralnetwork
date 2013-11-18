package com.neuralnetwork.xor;

public interface IActivationFunction
{
    public float apply(float v);

    public interface IDifferentiableFunction extends IActivationFunction
    {
        public float derivative(float v);
    }

    public class ThresholdFunction implements IActivationFunction
    {

        @Override
        public float apply(float v)
        {
            return v >= 0f ? 1f : 0f;
        }
    }

    public class SigmoidFunction implements IDifferentiableFunction
    {
        private final float slope;
        private final float negSlope;

        public SigmoidFunction(float slope)
        {
            this.slope = slope;
            this.negSlope = -slope;
        }

        @Override
        public float apply(float v)
        {
            return (float) (1f / (1f + Math.exp(negSlope*v)));
        }

        @Override
        public float derivative(float v)
        {
            float negSlopeV = negSlope * v;
            float denom = (float) (1f + Math.exp(negSlopeV));

            return (float) ( (slope * Math.exp(negSlopeV)) / denom * denom );

        }
    }
}
