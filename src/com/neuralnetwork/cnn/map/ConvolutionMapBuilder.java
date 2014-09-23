package com.neuralnetwork.cnn.map;

import com.neuralnetwork.cnn.filter.IConvolutionFilter;

public class ConvolutionMapBuilder extends AbstractFeatureMapBuilder<ConvolutionMapBuilder,ConvolutionMap>
{
    public ConvolutionMapBuilder setFilter(IConvolutionFilter... filter) {

        this.aFilters = filter;

        return this;
    }

    public IConvolutionFilter[] getFilter()
    {
        return (IConvolutionFilter[]) this.aFilters;
    }


    protected ConvolutionMap buildMap()
    {
        return new ConvolutionMap(this);
    }
}
