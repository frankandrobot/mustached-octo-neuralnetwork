package com.neuralnetwork.cnn.singlelayer;

import org.ejml.data.DenseMatrix64F;

public class SamplingSingleLayer extends AbstractCNNSingleLayer<SamplingSingleLayerBuilder>
{
    SamplingSingleLayer(SamplingSingleLayerBuilder builder) throws IllegalArgumentException
    {
        super(builder);
    }

    @Override
    protected void createFilters(SamplingSingleLayerBuilder builder)
    {
        aFilters = builder.getFilters();
    }

    @Override
    protected void createOutputMatrix(SamplingSingleLayerBuilder builder)
    {
        output = new DenseMatrix64F(
                inputDim/builder.kernelDim,
                inputDim/builder.kernelDim);
    }
}
