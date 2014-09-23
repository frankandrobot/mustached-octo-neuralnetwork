package com.neuralnetwork.cnn.map;

import com.neuralnetwork.cnn.filter.ISamplingFilter;
import com.neuralnetwork.core.neuron.Neuron;

public class SamplingMapBuilder extends AbstractCnnMapBuilder<SamplingMapBuilder,SamplingMap>
{
    public SamplingMapBuilder setFilter(ISamplingFilter... filter) {

        this.aFilters = filter;

        return this;
    }

    public ISamplingFilter[] getFilters()
    {
        return (ISamplingFilter[]) this.aFilters;
    }


    protected SamplingMap buildMap()
    {
        validate();

        return new SamplingMap(this);
    }

    private void validate()
    {
        if (inputDim % kernelDim != 0)
            throw new IllegalArgumentException("Input size must be a multiple of the kernel size");

        for(Neuron neuron:aSharedNeurons) {
            double[] weights = neuron.getWeightsWithoutBias();
            double weight = weights[0];

            for (int i = 1; i < weights.length; i++)
                if (weights[i] != weight)
                    throw new IllegalArgumentException("Kernel weights must all be equal");
        }
    }
}
