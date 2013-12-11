package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;

public class FeatureMap
{
    protected final int inputSize;
    protected double[][] aFeatureMap;

    protected final MapFunction mapFunction;
    protected final int receptiveFieldSize;

    protected final OutputClass outputClass;


    public FeatureMap(Builder builder) {
        if (builder.inputSize - builder.mapFunction.getReceptiveFieldSize() + 1 <= 0)
            throw new IllegalArgumentException("Receptive field size can't be larger than the input size");

        inputSize = builder.inputSize;

        mapFunction = builder.mapFunction;
        receptiveFieldSize = mapFunction.getReceptiveFieldSize();

        aFeatureMap = mapFunction.getFeatureMapCreator().createFeatureMap(inputSize);

        outputClass = new OutputClass();
    }

    static public class Builder
    {
        /**
         * The FeatureMap will operate on arrays of size #inputSize x #inputSize
         */
        private int inputSize;

        private MapFunction mapFunction;

        public Builder setInputSize(int inputSize)
        {
            this.inputSize = inputSize;
            return this;
        }

        public Builder setMapFunction(MapFunction mapFunction)
        {
            this.mapFunction = mapFunction;
            return this;
        }

    }

    protected class OutputClass
    {
        NVector neuronInput = new NVector().setSize(receptiveFieldSize * receptiveFieldSize + 1);

        public OutputClass()
        {
            neuronInput.set(neuronInput.size()-1, 1);
        }

        public void copy(final double[][]input, final int i, final int j)
        {
            int len=0;
            for(int a=i; a < i+receptiveFieldSize; a++)
                for(int b=j; b < j+receptiveFieldSize; b++)
                    neuronInput.set(len++, input[a][b]);
        }
    }

    static abstract protected class MapFunction
    {
        protected FeatureMapCreator featureMapCreator;

        abstract public double apply(NVector input);

        abstract public int getReceptiveFieldSize();

        public FeatureMapCreator getFeatureMapCreator()
        {
            return featureMapCreator;
        }
    }

    protected interface FeatureMapCreator
    {
        public double[][] createFeatureMap(int inputSize);

        public int getReceptiveFieldSize();
    }

    static public class Convolution extends MapFunction
    {
        protected Neuron sharedNeuron;

        public Convolution(Neuron neuron, int receptiveFieldSize)
        {
            if (receptiveFieldSize * receptiveFieldSize != neuron.getNumberOfWeights() - 1)
                throw new IllegalArgumentException("receptive field size squared must equal the number of weights minus the bias"
                        + "; otherwise it's not a receptive field");

            featureMapCreator = new ConvolutionFeatureMapCreator(receptiveFieldSize);
            sharedNeuron = neuron;
        }

        @Override
        public double apply(NVector input)
        {
            return sharedNeuron.output(input);
        }

        @Override
        public int getReceptiveFieldSize()
        {
            return featureMapCreator.getReceptiveFieldSize();
        }
    }

    static protected class ConvolutionFeatureMapCreator implements FeatureMapCreator
    {
        final protected int receptiveFieldSize;

        protected ConvolutionFeatureMapCreator(final int receptiveFieldSize)
        {
            this.receptiveFieldSize = receptiveFieldSize;
        }

        @Override
        public double[][] createFeatureMap(int inputSize)
        {
            final int n = inputSize - receptiveFieldSize + 1;
            return new double[n][n];

        }

        @Override
        public int getReceptiveFieldSize()
        {
            return receptiveFieldSize;
        }
    }

    public FeatureMap output(double[][] input)
    {
        for(int i=0; i<=input.length - receptiveFieldSize; i++)
            for(int j=0; j<=input[i].length - receptiveFieldSize; j++)
            {
                //copy over input into data struct
                outputClass.copy(input, i, j);
                //do it
                aFeatureMap[i][j] = mapFunction.apply(outputClass.neuronInput);
            }
        return this;
    }
}
