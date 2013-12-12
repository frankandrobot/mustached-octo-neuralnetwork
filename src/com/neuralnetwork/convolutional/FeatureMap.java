package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;

public class FeatureMap
{
    /**
     * The input array has dimensions #inputSize x #inputSize
     */
    protected final int inputSize;
    /**
     * The actual feature map. The dimensions depend on the MapFunction
     */
    protected double[][] aFeatureMap;

    /**
     * One of Convolution or subsampling
     */
    protected final MapFunction mapFunction;

    public FeatureMap(Builder builder) {
        if (builder.inputSize - builder.mapFunction.getReceptiveFieldSize() + 1 <= 0)
            throw new IllegalArgumentException("Receptive field size can't be larger than the input size");

        inputSize = builder.inputSize;

        mapFunction = builder.mapFunction;
        aFeatureMap = mapFunction.createFeatureMap(inputSize);
    }

    static public class Builder
    {
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

    /**
     * Convenience class
     */
    static protected class OutputClass
    {
        NVector mapInput;

        protected OutputClass() {}

        public void setMapInput(NVector mapInput)
        {
            this.mapInput = mapInput;
        }

        /**
         * Copies a chunk of size x size from the input starting at location i,j
         */
        public void copy(final double[][]input, final int size, final int i, final int j)
        {
            int len=0;
            for(int a=i; a < i+size; a++)
                for(int b=j; b < j+size; b++)
                    mapInput.set(len++, input[a][b]);
        }
    }

    static abstract protected class MapFunction
    {
        protected int receptiveFieldSize;
        protected OutputClass outputClass = new OutputClass();

        abstract protected double apply(NVector input);

        abstract public MapFunction setReceptiveFieldSize(int receptiveFieldSize);
        protected int getReceptiveFieldSize() { return receptiveFieldSize; }

        abstract protected double[][] createFeatureMap(int inputSize);

        abstract protected NVector generateMapInputCache();

        abstract protected void output(double[][] input, double[][] aFeatureMap);
    }

    static public class ConvolutionFunction extends MapFunction
    {
        protected Neuron sharedNeuron;

        public ConvolutionFunction(Neuron neuron)
        {
            sharedNeuron = neuron;
        }

        @Override
        public MapFunction setReceptiveFieldSize(int receptiveFieldSize)
        {
            if (receptiveFieldSize * receptiveFieldSize != sharedNeuron.getNumberOfWeights() - 1)
                throw new IllegalArgumentException("receptive field size squared must equal the number of weights minus the bias"
                        + "; otherwise it's not a receptive field");

            this.receptiveFieldSize = receptiveFieldSize;
            outputClass.setMapInput(generateMapInputCache());
            return this;
        }

        @Override
        protected double apply(NVector input)
        {
            return sharedNeuron.output(input);
        }

        @Override
        protected double[][] createFeatureMap(int inputSize)
        {
            final int n = inputSize - receptiveFieldSize + 1;
            return new double[n][n];

        }

        @Override
        protected NVector generateMapInputCache()
        {
            NVector neuronInput = new NVector().setSize(receptiveFieldSize * receptiveFieldSize + 1);
            neuronInput.set(neuronInput.size()-1, 1);
            return neuronInput;
        }

        @Override
        protected void output(double[][] input, double[][] aFeatureMap)
        {
            for(int i=0; i<=input.length - receptiveFieldSize; i++)
                for(int j=0; j<=input[i].length - receptiveFieldSize; j++)
                {
                    //copy over input into data struct
                    outputClass.copy(input, receptiveFieldSize, i, j);
                    //do it
                    aFeatureMap[i][j] = apply(outputClass.mapInput);
                }
        }
    }

    /**
     * Takes the average of the input of size #receptiveFieldSize x #receptiveFieldSize
     * then multiplies it by a scale factor, adds a bias, then applies an activation function
     */
    static public class SubSamplingFunction extends MapFunction
    {
        protected Neuron sharedNeuron;
        /**
         * For speed improvements
         */
        protected double oneOverInputSize;
        protected NVector neuronInput;

        public SubSamplingFunction(Neuron neuron)
        {
            sharedNeuron = neuron;
            neuronInput = new NVector().setSize(2);
            neuronInput.set(1, 1);
        }

        @Override
        protected double apply(NVector input)
        {
            double avg = oneOverInputSize * input.sumOfCoords();
            neuronInput.set(0, avg);
            return sharedNeuron.output(neuronInput);
        }

        @Override
        public MapFunction setReceptiveFieldSize(int receptiveFieldSize)
        {
            this.receptiveFieldSize = receptiveFieldSize;
            this.oneOverInputSize = 1.0 / (receptiveFieldSize * receptiveFieldSize);
            outputClass.setMapInput(generateMapInputCache());
            return this;
        }

        @Override
        protected double[][] createFeatureMap(int inputSize)
        {
            if (inputSize % receptiveFieldSize != 0)
                throw new IllegalArgumentException("input size must be a multiple of the receptive field size");
            final int n = inputSize / receptiveFieldSize;
            return new double[n][n];
        }

        @Override
        public NVector generateMapInputCache()
        {
            return new NVector().setSize(receptiveFieldSize * receptiveFieldSize);
        }

        public void output(double[][] input, double[][] aFeatureMap)
        {
            for(int i=0, smallI=0; i<=input.length - receptiveFieldSize; i+=receptiveFieldSize, smallI++)
                for(int j=0, smallJ=0; j<=input[i].length - receptiveFieldSize; j+=receptiveFieldSize, smallJ++)
                {
                    //copy over input into data struct
                    outputClass.copy(input, receptiveFieldSize, i, j);
                    //do it
                    aFeatureMap[smallI][smallJ] = apply(outputClass.mapInput);
                }
        }
    }

    public FeatureMap output(double[][] input)
    {
        mapFunction.output(input, aFeatureMap);
        return this;
    }
}
