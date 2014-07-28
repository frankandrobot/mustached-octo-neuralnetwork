package com.neuralnetwork.cnn.backprop;

import com.neuralnetwork.core.interfaces.OldINeuralNetwork;
import com.neuralnetwork.core.neuron.MNeuron;
import org.ejml.data.DenseMatrix64F;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

abstract public class OldFeatureMap implements OldINeuralNetwork.IMatrixNeuralNetwork
{
    /**
     * The input array has dimensions #inputSize x #inputSize
     */
    final protected int oneDimInputSize;
    /**
     * The actual feature map. The dimensions depend on the MapFunction
     */
    protected DenseMatrix64F mFeatureMap;
    final protected int numberNeurons;
    final protected MNeuron sharedNeuron;

    /**
     * Receptive field size is the size of the input of the neuron.
     * It should be a square.
     */
    final protected int receptiveFieldSize;
    final protected int sqrtReceptiveFieldSize;

    public OutputClass outputClass = new OutputClass();

    public OldFeatureMap(Builder builder)
    {
        if (builder.inputSize - builder.sqrtReceptiveFieldSize + 1 <= 0)
            throw new IllegalArgumentException("Receptive field size can't be larger than the input size");
        if (builder.sqrtReceptiveFieldSize * builder.sqrtReceptiveFieldSize != builder.receptiveFieldSize)
            throw new IllegalArgumentException();

        oneDimInputSize = builder.inputSize;
        sharedNeuron = builder.sharedNeuron;
        receptiveFieldSize = builder.receptiveFieldSize;
        sqrtReceptiveFieldSize = builder.sqrtReceptiveFieldSize;

        mFeatureMap = createFeatureMap(oneDimInputSize);

        numberNeurons = mFeatureMap.numCols * mFeatureMap.numRows;
    }

    static public class Builder
    {
        private int inputSize;
        private int receptiveFieldSize;
        private int sqrtReceptiveFieldSize;
        private MNeuron sharedNeuron;

        public Builder set1DInputSize(int inputSize)
        {
            this.inputSize = inputSize;
            return this;
        }

        public Builder setReceptiveFieldSize(int receptiveFieldSize)
        {
            this.receptiveFieldSize = receptiveFieldSize;
            this.sqrtReceptiveFieldSize = (int) Math.sqrt(receptiveFieldSize);
            return this;
        }

        public Builder setNeuron(MNeuron neuron)
        {
            this.sharedNeuron = neuron;
            return this;
        }

    }

    /**
     * Used by the constructor to create the feature map
     * @param inputSize 1D input size
     * @return feature map
     */
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
    abstract public void output(DenseMatrix64F input, DenseMatrix64F mFeatureMap);

    /**
     * Given:
     * - (i,j) representing a pixel in the input layer
     * - aWeightConnections - the matrix (in 1D form) of the receptive field of a pixel in feature map.
     *   A position in the matrix represents a weight. Its value is 1 iff that weight is connected to the
     *   input pixel. All values are initially 1.
     *
     * This method disables aWeightConnections.
     *
     * @param aWeightConnections see above
     * @param i pixel's row position in input
     * @param j pixel's col position in input
     */
    protected abstract void disableWeightConnections(int[] aWeightConnections, int i, int j);


    public DenseMatrix64F calculateFeatureMap(DenseMatrix64F input)
    {
        output(input, mFeatureMap);
        return mFeatureMap;
    }

    /**
     * @param input unlike a {@link com.neuralnetwork.core.deprecated.MultiLayerNetworkOld}, this is a square matrix.
     *              The implementation transforms it to a 1 x n matrix suitable for passing into a neuron
     * @return output
     */
    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F input)
    {
        calculateFeatureMap(input);
        return mFeatureMap;
    }

    @Override
    public DenseMatrix64F generateInducedLocalField(DenseMatrix64F input)
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
        return sharedNeuron;
    }

    public DenseMatrix64F getFeatureMap()
    {
        return mFeatureMap;
    }

    public int getInputDim()
    {
        throw new NotImplementedException();
    }

    @Override
    public String toString()
    {
        return getClass().getSimpleName();
    }

    /**
     * Given the weight in the receptive field and the pixel's col position in the input,
     * returns the pixel's col position in the feature map
     *
     * @param weight weight position in aWeightConnections
     * @param j pixel's col position in input
     * @return col position in feature map
     */
    abstract public int featureMapColPosition(int weight, int j);

    /**
     * Given the weight in the receptive field and the pixel's row position in the input,
     * returns the pixel's row position in the feature map
     *
     * @param weight weight position in aWeightConnections
     * @param i pixel's row position in input
     * @return row position in feature map
     */
    abstract public int featureMapRowPosition(int weight, int i);

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
                return sharedNeuron;
            }

            @Override
            public void remove()
            {
                throw new NotImplementedException();
            }
        };
    }

    /**
     * Convenience class
     */
    static protected class OutputClass
    {
        /**
         * Used for computations.
         * - passed directly to the neuron for #ConvolutionMapLayerOld
         */
        public DenseMatrix64F mapInput;

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
}
