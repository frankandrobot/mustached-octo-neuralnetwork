package com.neuralnetwork.cnn.layer.builder;

import com.neuralnetwork.cnn.filter.ISamplingFilter;
import com.neuralnetwork.cnn.layer.SamplingLayer;

public class SamplingLayerBuilder extends FeatureMapBuilder<SamplingLayerBuilder>
{
    public SamplingLayerBuilder setFilter(ISamplingFilter filter) {

        this.filter = filter;

        return this;
    }

    public ISamplingFilter getFilter()
    {
        return (ISamplingFilter) this.filter;
    }

    public SamplingLayer build()
    {
        return new SamplingLayer(this);
    }
}
