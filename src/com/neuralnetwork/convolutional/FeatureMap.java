package com.neuralnetwork.convolutional;

import org.ejml.data.DenseMatrix64F;

public class FeatureMap
{
    /**
     * The input array has dimensions #inputSize x #inputSize
     */
    protected final int inputSize;
    /**
     * The actual feature map. The dimensions depend on the MapFunction
     */
    protected DenseMatrix64F mFeatureMap;

    /**
     * One of Convolution or subsampling
     */
    protected final MapFunction mapFunction;

    public FeatureMap(Builder builder) {
        if (builder.inputSize - builder.mapFunction.sqrtReceptiveFieldSize + 1 <= 0)
            throw new IllegalArgumentException("Receptive field size can't be larger than the input size");

        inputSize = builder.inputSize;

        mapFunction = builder.mapFunction;
        mFeatureMap = mapFunction.createFeatureMap(inputSize);
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
        DenseMatrix64F mapInput;

        protected OutputClass() {}

        public void setMapInput(DenseMatrix64F mapInput)
        {
            this.mapInput = mapInput;
        }

        /**
         * Copies a chunk of size x size from the input starting at location i,j
         */
        public void copy(DenseMatrix64F input, final int size, final int i, final int j)
        {
            int len=0;
            for(int a=i; a < i+size; a++)
                for(int b=j; b < j+size; b++)
                    mapInput.set(0, len++, input.unsafe_get(a,b));
        }
    }

    static abstract protected class MapFunction
    {
        protected MNeuron sharedNeuron;
        /**
         * Receptive field size is the size of the input of the neuron.
         * It should be a square.
         */
        protected int receptiveFieldSize;
        protected int sqrtReceptiveFieldSize;
        protected OutputClass outputClass = new OutputClass();

        abstract protected double apply(DenseMatrix64F input);

        abstract public MapFunction setReceptiveFieldSize(int receptiveFieldSize);
        protected int getReceptiveFieldSize() { return receptiveFieldSize; }

        abstract protected DenseMatrix64F createFeatureMap(int inputSize);

        abstract protected DenseMatrix64F generateMapInputCache();

        abstract protected void output(DenseMatrix64F input, DenseMatrix64F aFeatureMap);
    }

    static public class ConvolutionFunction extends MapFunction
    {
        public ConvolutionFunction(MNeuron neuron)
        {
            sharedNeuron = neuron;
        }

        @Override
        public MapFunction setReceptiveFieldSize(int receptiveFieldSize)
        {
            if (receptiveFieldSize != sharedNeuron.getNumberOfWeights() - 1)
                throw new IllegalArgumentException("receptive field size must equal the number of weights minus the bias"
                        + "; otherwise it's not a receptive field");

            this.receptiveFieldSize = receptiveFieldSize;
            this.sqrtReceptiveFieldSize = (int) Math.sqrt(receptiveFieldSize);
            outputClass.setMapInput(generateMapInputCache());
            return this;
        }

        @Override
        protected double apply(DenseMatrix64F input)
        {
            return sharedNeuron.output(input);
        }

        @Override
        protected DenseMatrix64F createFeatureMap(int inputSize)
        {
            final int n = inputSize - sqrtReceptiveFieldSize + 1;
            return new DenseMatrix64F(n,n);

        }

        @Override
        protected DenseMatrix64F generateMapInputCache()
        {
            DenseMatrix64F neuronInput = new DenseMatrix64F(1, receptiveFieldSize);
            return neuronInput;
        }

        @Override
        protected void output(DenseMatrix64F input, DenseMatrix64F aFeatureMap)
        {
            for(int i=0; i<=input.numRows - sqrtReceptiveFieldSize; i++)
                for(int j=0; j<=input.numCols - sqrtReceptiveFieldSize; j++)
                {
                    //copy over input into data struct
                    outputClass.copy(input, sqrtReceptiveFieldSize, i, j);
                    //do it
                    aFeatureMap.set(i,j, apply(outputClass.mapInput));
                }
        }
    }

    /**
     * Takes the average of the input of size #sqrtReceptiveFieldSize x #sqrtReceptiveFieldSize
     * then multiplies it by a scale factor, adds a bias, then applies an activation function
     */
    /*static public class SubSamplingFunction extends MapFunction
    {
        public SubSamplingFunction(MNeuron neuron)
        {
            if (neuron.getNumberOfWeights() != 2)
                throw new IllegalArgumentException(SubSamplingFunction.class.getSimpleName()+" needs exactly 2 weights");

            sharedNeuron = neuron;
        }

        @Override
        protected double apply(DenseMatrix64F input)
        {
            neuronInput.set(0, input.sumOfCoords());
            return sharedNeuron.output(neuronInput);
        }

        @Override
        public MapFunction setReceptiveFieldSize(int receptiveFieldSize)
        {
            this.receptiveFieldSize = receptiveFieldSize;
            this.sqrtReceptiveFieldSize = (int) Math.sqrt(receptiveFieldSize);
            outputClass.setMapInput(generateMapInputCache());
            return this;
        }

        @Override
        protected DenseMatrix64F createFeatureMap(int inputSize)
        {
            if (inputSize % sqrtReceptiveFieldSize != 0)
                throw new IllegalArgumentException("input size must be a multiple of the receptive field size");
            final int n = inputSize / sqrtReceptiveFieldSize;
            return new DenseMatrix64F(n,n);
        }

        @Override
        public NVector generateMapInputCache()
        {
            return new NVector().setSize(receptiveFieldSize);
        }

        public void output(DenseMatrix64F input, DenseMatrix64F aFeatureMap)
        {
            for(int i=0, smallI=0; i<=input.length - sqrtReceptiveFieldSize; i+=sqrtReceptiveFieldSize, smallI++)
                for(int j=0, smallJ=0; j<=input[i].length - sqrtReceptiveFieldSize; j+=sqrtReceptiveFieldSize, smallJ++)
                {
                    //copy over input into data struct
                    outputClass.copy(input, sqrtReceptiveFieldSize, i, j);
                    //do it
                    aFeatureMap[smallI][smallJ] = apply(outputClass.mapInput);
                }
        }
    }*/

    public DenseMatrix64F output(DenseMatrix64F input)
    {
        mapFunction.output(input, mFeatureMap);
        return mFeatureMap;
    }

    public DenseMatrix64F getFeatureMap()
    {
        return mFeatureMap;
    }
}
