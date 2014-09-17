package com.neuralnetwork.cnn.layer.builder;

import com.neuralnetwork.cnn.filter.ISamplingFilter;
import com.neuralnetwork.cnn.layer.SamplingMap;

public class SamplingMapBuilder extends AbstractFeatureMapBuilder<SamplingMapBuilder>
{
    public SamplingMapBuilder setFilter(ISamplingFilter filter) {

        this.filter = filter;

        return this;
    }

    public ISamplingFilter getFilter()
    {
        return (ISamplingFilter) this.filter;
    }

    public SamplingMap build()
    {
        return new SamplingMap(this);
    }
}
