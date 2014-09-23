package com.neuralnetwork.cnn.map;

import org.ejml.data.DenseMatrix64F;

public class SamplingMap extends AbstractCnnMap
{
    SamplingMap(SamplingMapBuilder builder) throws IllegalArgumentException
    {
        //extract from the builder
        aSharedNeurons = builder.aSharedNeurons;
        inputDim = builder.inputDim;
        filter = builder.getFilters();
        aKernels = builder.aKernels;

        output = new DenseMatrix64F(
                inputDim/builder.kernelDim,
                inputDim/builder.kernelDim);
    }
}
