package com.neuralnetwork.cnn.singlelayer;

import com.neuralnetwork.cnn.filter.IConvolutionFilter;

public class ConvolutionSingleLayerBuilder extends AbstractCNNSingleLayerBuilder<ConvolutionSingleLayerBuilder,ConvolutionSingleLayer>
{
    public ConvolutionSingleLayerBuilder setFilter(IConvolutionFilter... filter) {

        this.aFilters = filter;

        return this;
    }

    public IConvolutionFilter[] getFilter()
    {
        return (IConvolutionFilter[]) this.aFilters;
    }


    protected ConvolutionSingleLayer buildMap()
    {
        return new ConvolutionSingleLayer(this);
    }
}
