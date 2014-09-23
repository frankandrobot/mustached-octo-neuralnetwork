package com.neuralnetwork.cnn.map;

import org.ejml.data.DenseMatrix64F;

public class ConvolutionMap extends AbstractCnnMap
{
    ConvolutionMap(ConvolutionMapBuilder builder) throws IllegalArgumentException
    {
        //extract from builder
        aSharedNeurons = builder.aSharedNeurons;
        inputDim = builder.inputDim;
        aKernels = builder.aKernels;
        filter = builder.getFilter();

        int kernelDim = (int) Math.sqrt(aSharedNeurons[0].getNumberOfWeights() - 1);

        output = new DenseMatrix64F(
                inputDim-kernelDim+1,
                inputDim-kernelDim+1);
    }
}
