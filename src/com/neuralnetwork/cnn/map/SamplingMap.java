package com.neuralnetwork.cnn.map;

import org.ejml.data.DenseMatrix64F;

public class SamplingMap extends AbstractCnnMap<SamplingMapBuilder>
{
    SamplingMap(SamplingMapBuilder builder) throws IllegalArgumentException
    {
        super(builder);
    }

    @Override
    protected void createFilters(SamplingMapBuilder builder)
    {
        aFilters = builder.getFilters();
    }

    @Override
    protected void createOutputMatrix(SamplingMapBuilder builder)
    {
        output = new DenseMatrix64F(
                inputDim/builder.kernelDim,
                inputDim/builder.kernelDim);
    }
}
