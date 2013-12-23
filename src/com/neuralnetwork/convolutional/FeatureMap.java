package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

abstract public class FeatureMap implements INeuralNetwork.IMatrixNeuralNetwork
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

    protected OutputClass outputClass = new OutputClass();

    public FeatureMap(Builder builder) {
        if (builder.inputSize - builder.sqrtReceptiveFieldSize + 1 <= 0)
            throw new IllegalArgumentException("Receptive field size can't be larger than the input size");
        else if (builder.sqrtReceptiveFieldSize * builder.sqrtReceptiveFieldSize != builder.receptiveFieldSize)
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
     *   input pixel.
     *
     * This method populates aWeightConnections.
     *
     * @param aWeightConnections see above
     * @param i pixel's row position in input
     * @param j pixel's col position in input
     */
    public abstract void calculateWeightConnections(int[] aWeightConnections, int i, int j);


    /**
     * Convenience class
     */
    static protected class OutputClass
    {
        /**
         * Used for computations.
         * - passed directly to the neuron for #ConvolutionMap
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

    public DenseMatrix64F calculateFeatureMap(DenseMatrix64F input)
    {
        output(input, mFeatureMap);
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
        return sharedNeuron;
    }

    public DenseMatrix64F getFeatureMap()
    {
        return mFeatureMap;
    }


    @Override
    public String toString()
    {
        return getClass().getSimpleName();
    }

    /**
     * Gets the input pixels col position in the feature map for the given weight
     *
     * @param weight weight position in aWeightConnections
     * @param j pixel's col position in input
     * @return
     */
    public int featureMapColPosition(int weight, int j)
    {
        return featureMapColPosition(weight, j);
    }

    /**
     * Gets the input pixel's row position in the feature map for the given weight
     *
     * @param weight weight position in aWeightConnections
     * @param i pixel's row position in input
     * @return
     */
    public int featureMapRowPosition(int weight, int i)
    {
        return featureMapRowPosition(weight, i);
    }

    static public class ConvolutionMap extends FeatureMap
    {

        public ConvolutionMap(Builder builder)
        {
            super(builder);

            outputClass.setMapInput(new DenseMatrix64F(1, receptiveFieldSize));
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
        public void output(DenseMatrix64F input, DenseMatrix64F aFeatureMap)
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

        @Override
        public void calculateWeightConnections(int[] aWeightConnections, int i, int j)
        {
            //find distance to borders
            final int distanceToL = j;
            final int distanceToT = i;
            final int distanceToR = oneDimInputSize - j;
            final int idstanceToB = oneDimInputSize - i;

            //reset weights
            for(int w=0; w<aWeightConnections.length; w++)
                aWeightConnections[w] = 1;

            //disable from left border
            for(int col=distanceToL+1; col<sqrtReceptiveFieldSize; col++)
                disableCol(aWeightConnections, col);
            //disable from right border
            for(int col=distanceToR+1; col<sqrtReceptiveFieldSize; col++ )
                disableCol(aWeightConnections, col);
            //disable left border
            for(int row=distanceToT; row<)
        }
    }

    /**
     * Takes the average of the input of size #sqrtReceptiveFieldSize x #sqrtReceptiveFieldSize
     * then multiplies it by a scale factor, adds a bias, then applies an activation function
     */
    static public class SubSamplingMap extends FeatureMap
    {
        DenseMatrix64F neuronInput = new DenseMatrix64F(1,1);

        public SubSamplingMap(Builder builder)
        {
            super(builder);

            if (sharedNeuron.getNumberOfWeights() != 2)
                throw new IllegalArgumentException(SubSamplingMap.class.getSimpleName()+" needs exactly 2 weights");
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

        @Override
        public void calculateWeightConnections(int[] aWeightConnections, int i, int j) {
            //To change body of implemented methods use File | Settings | File Templates.
        }
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
                return sharedNeuron;
            }

            @Override
            public void remove()
            {
                throw new NotImplementedException();
            }
        };
    }
}
