package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;

public class FeatureMap
{
    protected double[][] aFeatureMap;

    protected final int inputSize;

    protected Neuron sharedNeuron;
    protected final int receptiveFieldSize;

    protected final OutputClass outputClass;

    public FeatureMap(Builder builder)
    {
        if (builder.inputSize - builder.receptiveFieldSize <= 0)
            throw new IllegalArgumentException("Receptive field size can't be larger than the input size");

        inputSize = builder.inputSize;

        receptiveFieldSize = builder.receptiveFieldSize;
        sharedNeuron = builder.sharedNeuron;

        final int n = inputSize - receptiveFieldSize;
        aFeatureMap = new double[n][n];

        outputClass = new OutputClass();
    }

    static public class Builder
    {
        /**
         * The FeatureMap will operate on arrays of size #inputSize x #inputSize
         */
        private int inputSize;
        /**
         * Each neuron will be responsible for #receptiveFieldSize x #receptiveFieldSize square in the input
         */
        private int receptiveFieldSize;
        private Neuron sharedNeuron;

        public void setInputSize(int inputSize)
        {
            this.inputSize = inputSize;
        }

        public void setReceptiveFieldSize(int receptiveFieldSize)
        {
            this.receptiveFieldSize = receptiveFieldSize;
        }

        public void setSharedNeuron(Neuron neuron)
        {
            this.sharedNeuron = neuron;
        }
    }

    protected class OutputClass
    {
        NVector neuronInput = new NVector().setSize(receptiveFieldSize * receptiveFieldSize);

        public void copy(final double[][]input, final int i, final int j)
        {
            int len=0;
            for(int a=i; a< receptiveFieldSize; a++)
                for(int b=j; b< receptiveFieldSize; b++)
                    neuronInput.set(len++, input[a][b]);
        }

        protected void convolve(final int i, final int j)
        {
            aFeatureMap[i][j] = sharedNeuron.output(neuronInput);
        }
    }

    public FeatureMap output(double[][] input)
    {
        for(int i=0; i<input.length - receptiveFieldSize; i++)
            for(int j=0; j<input[i].length - receptiveFieldSize; j++)
            {
                //copy over input into data struct
                outputClass.copy(input, i, j);
                //do it
                outputClass.convolve(i, j);
            }
        return this;
    }
}
