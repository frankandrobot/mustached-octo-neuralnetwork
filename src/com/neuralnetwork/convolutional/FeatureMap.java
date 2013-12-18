package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

public class FeatureMap implements INeuralNetwork.IMatrixNeuralNetwork
{
    /**
     * The input array has dimensions #inputSize x #inputSize
     */
    protected final int inputSize;
    /**
     * The actual feature map. The dimensions depend on the MapFunction
     */
    protected DenseMatrix64F mFeatureMap;
    final protected int numberNeurons;

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

        numberNeurons = mFeatureMap.numCols * mFeatureMap.numRows;
    }

    @Override
    public Iterator<MNeuron> iterator()
    {
        return new Iterator<MNeuron>() {
            int len = 0;
            @Override
            public boolean hasNext()
            {
                return len < numberNeurons;
            }

            @Override
            public MNeuron next()
            {
                ++len;
                return mapFunction.sharedNeuron;
            }

            @Override
            public void remove()
            {
                throw new NotImplementedException();
            }
        };
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
        /**
         * Used for computations.
         * - passed directly to the neuron for #ConvolutionFunction
         */
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
                    mapInput.unsafe_set(0, len++, input.unsafe_get(a,b));
        }

        /**
         * Sums over the elements in a chunk of size x size from the input starting at location i,j
         */
        public double elementSum(DenseMatrix64F input, final int size, final int i, final int j)
        {
            double len=0;
            for(int a=i; a < i+size; a++)
                for(int b=j; b < j+size; b++)
                    len += input.unsafe_get(a,b);
            return len;
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

        abstract public MapFunction setReceptiveFieldSize(int receptiveFieldSize);
        protected int getReceptiveFieldSize() { return receptiveFieldSize; }

        abstract protected DenseMatrix64F createFeatureMap(int inputSize);

        /**
         * Calculates the induced local field at (x,y) in the feature map
         *
         * @return induced local field at (x,y)
         */
        abstract public double rawoutput(final DenseMatrix64F input, final int x, final int y);

        /**
         * Calculates the value of the activation function at (x,y) in the feature map for the given input
         *
         * @return value of activation function at (x,y)
         */
        abstract public double output(final DenseMatrix64F input, final int x, final int y);

        /**
         * Calculates the value of the activation function over all (x,y) in the feature map
         * *and* saves it into the feature map
         */
        abstract protected void output(DenseMatrix64F input, DenseMatrix64F mFeatureMap);
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
            outputClass.setMapInput(new DenseMatrix64F(1, receptiveFieldSize));
            return this;
        }

        @Override
        protected DenseMatrix64F createFeatureMap(int inputSize)
        {
            final int n = inputSize - sqrtReceptiveFieldSize + 1;
            return new DenseMatrix64F(n,n);

        }

        @Override
        public double output(DenseMatrix64F input, int x, int y)
        {
            double inducedLocalField = rawoutput(input, x, y);
            return sharedNeuron.phi().apply(inducedLocalField);
        }

        @Override
        public double rawoutput(final DenseMatrix64F input, final int x, final int y)
        {
            outputClass.copy(input, sqrtReceptiveFieldSize, x, y);
            return sharedNeuron.rawoutput(outputClass.mapInput);
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
                    aFeatureMap.unsafe_set(i, j, sharedNeuron.output(outputClass.mapInput));
                }
        }
    }

    /**
     * Takes the average of the input of size #sqrtReceptiveFieldSize x #sqrtReceptiveFieldSize
     * then multiplies it by a scale factor, adds a bias, then applies an activation function
     */
    static public class SubSamplingFunction extends MapFunction
    {
        DenseMatrix64F neuronInput = new DenseMatrix64F(1,1);

        public SubSamplingFunction(MNeuron neuron)
        {
            if (neuron.getNumberOfWeights() != 2)
                throw new IllegalArgumentException(SubSamplingFunction.class.getSimpleName()+" needs exactly 2 weights");

            sharedNeuron = neuron;
        }

        @Override
        public MapFunction setReceptiveFieldSize(int receptiveFieldSize)
        {
            this.receptiveFieldSize = receptiveFieldSize;
            this.sqrtReceptiveFieldSize = (int) Math.sqrt(receptiveFieldSize);
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
        public double output(DenseMatrix64F input, int x, int y)
        {
            double inducedLocalField = rawoutput(input, x, y);
            return sharedNeuron.phi().apply(inducedLocalField);
        }

        @Override
        public double rawoutput(final DenseMatrix64F input, final int x, final int y)
        {
            double sum = outputClass.elementSum(input,
                                                sqrtReceptiveFieldSize,
                                                x*sqrtReceptiveFieldSize,
                                                y*sqrtReceptiveFieldSize);
            neuronInput.unsafe_set(0,0,sum);
            return sharedNeuron.rawoutput(neuronInput);
        }

        public void output(DenseMatrix64F input, DenseMatrix64F aFeatureMap)
        {
            for(int i=0, smallI=0; i<=input.numRows - sqrtReceptiveFieldSize; i+=sqrtReceptiveFieldSize, smallI++)
                for(int j=0, smallJ=0; j<=input.numCols - sqrtReceptiveFieldSize; j+=sqrtReceptiveFieldSize, smallJ++)
                {
                    //copy over input into data struct
                    double sum = outputClass.elementSum(input, sqrtReceptiveFieldSize, i, j);
                    neuronInput.unsafe_set(0,0,sum);
                    //do it
                    aFeatureMap.unsafe_set(smallI,smallJ, sharedNeuron.output(neuronInput));
                }
        }
    }

    public DenseMatrix64F calculateFeatureMap(DenseMatrix64F input)
    {
        mapFunction.output(input, mFeatureMap);
        return mFeatureMap;
    }

    /**
     * @param input unlike a {@link com.neuralnetwork.core.MultiLayerNetwork}, this is a square matrix.
     *              The implementation transforms it to a 1 x n matrix suitable for passing into a neuron
     * @return output
     */
    @Override
    public DenseMatrix64F output(DenseMatrix64F input)
    {
        calculateFeatureMap(input);
        return mFeatureMap;
    }

    @Override
    public DenseMatrix64F inducedLocalField(DenseMatrix64F input)
    {
        throw new NotImplementedException();
    }

    @Override
    public int getNumberOfNeurons()
    {
        return numberNeurons;
    }

    @Override
    public MNeuron getNeuron(int neuron)
    {
        return mapFunction.sharedNeuron;
    }

    public DenseMatrix64F getFeatureMap()
    {
        return mFeatureMap;
    }

    /**
     * Calculates the induced local field at (x,y) in the feature map
     *
     * @return induced local field at (x,y)
     */
    public double rawoutput(final DenseMatrix64F input, final int x, final int y)
    {
        return mapFunction.rawoutput(input,x,y);
    }

    /**
     * Calculates the value of the activation function at (x,y) in the feature map for the given input
     *
     * @return value of activation function at (x,y)
     */
    public double output(final DenseMatrix64F input, final int x, final int y)
    {
        return mapFunction.output(input,x,y);
    }
}
