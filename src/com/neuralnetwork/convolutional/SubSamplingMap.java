package com.neuralnetwork.convolutional;

import org.ejml.data.DenseMatrix64F;

import java.util.Arrays;

/**
 * Takes the average of the input of size #sqrtReceptiveFieldSize x #sqrtReceptiveFieldSize
 * then multiplies it by a scale factor, adds a bias, then applies an activation function
 */
public class SubSamplingMap extends FeatureMap
{
    DenseMatrix64F neuronInput = new DenseMatrix64F(1,1);

    public SubSamplingMap(Builder builder)
    {
        super(builder);

        if (sharedNeuron.getNumberOfWeights() != 2)
            throw new IllegalArgumentException(com.neuralnetwork.convolutional.SubSamplingMap.class.getSimpleName()+" needs exactly 2 weights");
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
    public void disableWeightConnections(int[] aWeightConnections, int i, int j)
    {
        Arrays.fill(aWeightConnections, 0);
        aWeightConnections[0] = 1;
    }

    @Override
    public int featureMapColPosition(int weight, int j)
    {
        return j / sqrtReceptiveFieldSize;
    }

    @Override
    public int featureMapRowPosition(int weight, int i)
    {
        return i / sqrtReceptiveFieldSize;
    }
}
