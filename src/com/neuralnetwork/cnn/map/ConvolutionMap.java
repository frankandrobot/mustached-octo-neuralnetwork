package com.neuralnetwork.cnn.map;

import org.ejml.data.DenseMatrix64F;

public class ConvolutionMap extends AbstractCnnMap<ConvolutionMapBuilder>
{
    public ConvolutionMap(ConvolutionMapBuilder builder)
    {
        super(builder);
    }

    @Override
    protected void createFilters(ConvolutionMapBuilder builder)
    {
        aFilters = builder.getFilter();
    }

    @Override
    protected void createOutputMatrix(ConvolutionMapBuilder builder)
    {
        output = new DenseMatrix64F(
                inputDim-builder.kernelDim+1,
                inputDim-builder.kernelDim+1);
    }
}
