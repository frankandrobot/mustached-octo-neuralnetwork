package com.neuralnetwork.cnn.singlelayer;

import org.ejml.data.DenseMatrix64F;

public class ConvolutionSingleLayer extends AbstractCNNSingleLayer<ConvolutionSingleLayerBuilder>
{
    public ConvolutionSingleLayer(ConvolutionSingleLayerBuilder builder)
    {
        super(builder);
    }

    @Override
    protected void createFilters(ConvolutionSingleLayerBuilder builder)
    {
        aFilters = builder.getFilter();
    }

    @Override
    protected void createOutputMatrix(ConvolutionSingleLayerBuilder builder)
    {
        output = new DenseMatrix64F(
                inputDim-builder.kernelDim+1,
                inputDim-builder.kernelDim+1);
    }
}
