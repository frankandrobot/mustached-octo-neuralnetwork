package com.neuralnetwork.cnn.map.builder;

import com.neuralnetwork.cnn.filter.IConvolutionFilter;
import com.neuralnetwork.cnn.map.ConvolutionMap;

public class ConvolutionMapBuilder extends AbstractFeatureMapBuilder<ConvolutionMapBuilder>
{
    public ConvolutionMapBuilder setFilter(IConvolutionFilter filter) {

        this.filter = filter;

        return this;
    }

    public IConvolutionFilter getFilter()
    {
        return (IConvolutionFilter) this.filter;
    }

    public ConvolutionMap build()
    {
        return new ConvolutionMap(this);
    }
}
