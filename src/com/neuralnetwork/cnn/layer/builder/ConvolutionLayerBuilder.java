package com.neuralnetwork.cnn.layer.builder;

import com.neuralnetwork.cnn.filter.IConvolutionFilter;
import com.neuralnetwork.cnn.layer.ConvolutionLayer;

public class ConvolutionLayerBuilder extends FeatureMapBuilder<ConvolutionLayerBuilder>
{
    public ConvolutionLayerBuilder setFilter(IConvolutionFilter filter) {

        this.filter = filter;

        return this;
    }

    public IConvolutionFilter getFilter()
    {
        return (IConvolutionFilter) this.filter;
    }

    public ConvolutionLayer build()
    {
        return new ConvolutionLayer(this);
    }
}
